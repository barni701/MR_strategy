from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_TICKERS = ["SPY", "AAPL", "DIA", "IWM"]
FEATURE_COLUMNS = [
    "SMA_200",
    "SMA_5",
    "RSI_2",
    "ret_1",
    "ret_3",
    "ret_5",
    "recent_vol_20",
    "oc_ret",
    "hl_range",
    "SMA200_Dist",
    "SMA5_Dist",
    "Trade_Direction",
    "overnight_ret",
]
DEFAULT_DATA_DIR = Path("Data_1D")
DEFAULT_OUTPUT_DIR = Path("ML_Data") / "lstm_daily"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
DEFAULT_PATIENCE = 15
GRAD_CLIP_NORM = 1.0
EPS = 1e-6


@dataclass
class SplitDataset:
    sequences: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame


@dataclass
class DatasetBundle:
    full_sequences: np.ndarray
    full_labels: np.ndarray
    full_metadata: pd.DataFrame
    train: SplitDataset
    validation: SplitDataset
    test: SplitDataset
    split_info: dict
    skipped_tickers: list[str]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class LSTMWinPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(inputs)
        final_hidden = hidden_state[-1]
        return self.classifier(self.dropout(final_hidden)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a daily multi-ticker LSTM to estimate trade win probability."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--label-threshold", type=float, default=0.003)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--rsi-os", type=float, default=60.0)
    parser.add_argument("--rsi-ob", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values, dtype=float)
    positive_mask = values >= 0
    negative_mask = ~positive_mask
    output[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[negative_mask])
    output[negative_mask] = exp_values / (1.0 + exp_values)
    return output


def timestamp_to_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def load_daily_ohlcv(data_dir: Path, ticker: str) -> pd.DataFrame | None:
    path = data_dir / f"{ticker}_1d.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    daily = (
        df.resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    daily["symbol"] = ticker
    return daily


def compute_rsi2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["SMA_200"] = data["close"].rolling(window=200).mean()
    data["SMA_5"] = data["close"].rolling(window=5).mean()

    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 2, adjust=False).mean()
    rs = avg_gain / avg_loss
    data["RSI_2"] = 100 - (100 / (1 + rs))

    data["ret_1"] = data["close"].pct_change()
    data["ret_3"] = data["close"].pct_change(3)
    data["ret_5"] = data["close"].pct_change(5)
    data["oc_ret"] = (data["close"] - data["open"]) / data["open"]
    data["hl_range"] = (data["high"] - data["low"]) / data["open"]
    data["recent_vol_20"] = data["ret_1"].rolling(20).std().shift(1)
    data["overnight_ret"] = (data["open"] / data["close"].shift(1)) - 1
    data["SMA200_Dist"] = (data["close"] - data["SMA_200"]) / data["SMA_200"]
    data["SMA5_Dist"] = (data["close"] - data["SMA_5"]) / data["SMA_5"]
    return data


def drop_invalid_indicator_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["SMA_200", "SMA_5", "RSI_2"]
    return df.dropna(subset=required_columns).copy()


def generate_rsi2_signals(
    df: pd.DataFrame,
    trade_direction: str = "long_only",
    rsi_os: float = 60.0,
    rsi_ob: float = 40.0,
) -> pd.DataFrame:
    data = df.copy()
    above_200 = data["close"] > data["SMA_200"]
    below_200 = data["close"] < data["SMA_200"]
    oversold = data["RSI_2"] < rsi_os
    overbought = data["RSI_2"] > rsi_ob

    long_entry = above_200 & oversold
    short_entry = below_200 & overbought
    long_exit = data["close"] > data["SMA_5"]
    short_exit = data["close"] < data["SMA_5"]

    data["Trade_Direction"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))

    long_entry_np = long_entry.fillna(False).to_numpy()
    short_entry_np = short_entry.fillna(False).to_numpy()
    long_exit_np = long_exit.fillna(False).to_numpy()
    short_exit_np = short_exit.fillna(False).to_numpy()

    position_state = np.zeros(len(data), dtype=np.int8)
    signal_action = np.zeros(len(data), dtype=np.int8)
    entry_signal = np.zeros(len(data), dtype=np.int8)
    exit_signal = np.zeros(len(data), dtype=np.int8)

    in_position = 0
    for index in range(len(data)):
        if in_position == 1:
            if long_exit_np[index]:
                signal_action[index] = -1
                exit_signal[index] = 1
                in_position = 0
        elif in_position == -1:
            if short_exit_np[index]:
                signal_action[index] = 1
                exit_signal[index] = 1
                in_position = 0
        else:
            if trade_direction in {"long_only", "both"} and long_entry_np[index]:
                signal_action[index] = 1
                entry_signal[index] = 1
                in_position = 1
            elif trade_direction in {"short_only", "both"} and short_entry_np[index]:
                signal_action[index] = -1
                entry_signal[index] = 1
                in_position = -1

        position_state[index] = in_position

    data["Entry_Signal"] = entry_signal
    data["Exit_Signal"] = exit_signal
    data["Signal_Action"] = signal_action
    data["Position_State"] = position_state
    return data


def simulate_trade_from_entry(
    df: pd.DataFrame,
    entry_dt: pd.Timestamp,
    signal_exits: pd.DatetimeIndex,
    cost_bps: float,
    direction: int,
) -> dict | None:
    entry_index = df.index.get_loc(entry_dt)
    entry_price = float(df.loc[entry_dt, "open"])
    next_exits = signal_exits[signal_exits > entry_dt]

    if len(next_exits) == 0:
        return None

    exit_dt = next_exits[0]
    exit_index = df.index.get_loc(exit_dt)
    exit_price = float(df.iloc[exit_index]["open"])

    if direction == 1:
        gross_return = (exit_price - entry_price) / entry_price
    else:
        gross_return = (entry_price - exit_price) / entry_price

    net_return = gross_return - (2 * cost_bps * 1e-4)
    return {
        "Entry_Time": entry_dt,
        "Exit_Time": exit_dt,
        "Direction": int(direction),
        "Entry_Price": entry_price,
        "Exit_Price": exit_price,
        "Holding_Bars": int(exit_index - entry_index),
        "Exit_Reason": "SMA_5 Signal Exit",
        "Gross_Return": float(gross_return),
        "Net_Return": float(net_return),
    }


def build_trade_sequences(
    df_with_signals: pd.DataFrame,
    ticker: str,
    lookback: int,
    feature_columns: list[str],
    cost_bps: float,
    label_threshold: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = df_with_signals.copy()
    df["Exec_Entry"] = df["Entry_Signal"].shift(1).fillna(0).astype(np.int8)
    df["Exec_Exit"] = df["Exit_Signal"].shift(1).fillna(0).astype(np.int8)
    df["Exec_Direction"] = df["Signal_Action"].shift(1).fillna(0).astype(np.int8)

    entry_times = df.index[df["Exec_Entry"] == 1]
    signal_exits = df.index[df["Exec_Exit"] == 1]

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict] = []

    for entry_dt in entry_times:
        entry_index = df.index.get_loc(entry_dt)
        signal_index = entry_index - 1
        if signal_index < 0:
            continue

        signal_dt = df.index[signal_index]
        direction = int(df.loc[entry_dt, "Exec_Direction"])
        if direction == 0:
            continue

        trade = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps, direction)
        if trade is None:
            continue

        start_index = signal_index - lookback + 1
        if start_index < 0:
            continue

        feature_window = df.iloc[start_index : signal_index + 1][feature_columns]
        if len(feature_window) != lookback:
            continue

        values = feature_window.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            continue

        label = int(trade["Net_Return"] > label_threshold)
        sequences.append(values)
        labels.append(label)
        rows.append(
            {
                "Ticker": ticker,
                "Signal_Time": signal_dt,
                "Entry_Time": trade["Entry_Time"],
                "Exit_Time": trade["Exit_Time"],
                "Direction": trade["Direction"],
                "Entry_Price": trade["Entry_Price"],
                "Exit_Price": trade["Exit_Price"],
                "Holding_Bars": trade["Holding_Bars"],
                "Exit_Reason": trade["Exit_Reason"],
                "Gross_Return": trade["Gross_Return"],
                "Net_Return": trade["Net_Return"],
                "y_good": label,
            }
        )

    if not sequences:
        empty_sequences = np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.float32)
        empty_meta = pd.DataFrame(columns=["Ticker", "Signal_Time", "Entry_Time", "Exit_Time", "y_good"])
        return empty_sequences, empty_labels, empty_meta

    metadata = pd.DataFrame(rows)
    metadata["Signal_Time"] = pd.to_datetime(metadata["Signal_Time"], utc=True)
    metadata["Entry_Time"] = pd.to_datetime(metadata["Entry_Time"], utc=True)
    metadata["Exit_Time"] = pd.to_datetime(metadata["Exit_Time"], utc=True)
    return np.stack(sequences), np.asarray(labels, dtype=np.float32), metadata


def split_by_signal_time(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    unique_times = pd.DatetimeIndex(metadata["Signal_Time"].sort_values().unique())
    if len(unique_times) < 3:
        raise ValueError("Need at least 3 unique signal dates to create train/validation/test splits.")

    train_count = max(1, int(len(unique_times) * TRAIN_RATIO))
    val_count = max(1, int(len(unique_times) * VAL_RATIO))

    if train_count + val_count >= len(unique_times):
        train_count = max(1, len(unique_times) - 2)
        val_count = 1

    train_times = unique_times[:train_count]
    val_times = unique_times[train_count : train_count + val_count]
    test_times = unique_times[train_count + val_count :]

    if len(train_times) == 0 or len(val_times) == 0 or len(test_times) == 0:
        raise ValueError("Chronological split produced an empty train, validation, or test segment.")

    signal_times = metadata["Signal_Time"]
    return {
        "train": signal_times.isin(train_times).to_numpy(),
        "validation": signal_times.isin(val_times).to_numpy(),
        "test": signal_times.isin(test_times).to_numpy(),
        "train_start": train_times[0],
        "train_end": train_times[-1],
        "validation_start": val_times[0],
        "validation_end": val_times[-1],
        "test_start": test_times[0],
        "test_end": test_times[-1],
    }


def build_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    all_sequences: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_metadata: list[pd.DataFrame] = []
    skipped_tickers: list[str] = []

    for ticker in args.tickers:
        daily_data = load_daily_ohlcv(args.data_dir, ticker)
        if daily_data is None:
            skipped_tickers.append(f"{ticker}: missing data file")
            print(f"Skipping {ticker}: data file not found in {args.data_dir}")
            continue

        processed = compute_rsi2_indicators(daily_data)
        processed = drop_invalid_indicator_rows(processed)
        processed = generate_rsi2_signals(
            processed,
            trade_direction="long_only",
            rsi_os=args.rsi_os,
            rsi_ob=args.rsi_ob,
        )

        ticker_sequences, ticker_labels, ticker_metadata = build_trade_sequences(
            processed,
            ticker=ticker,
            lookback=args.lookback,
            feature_columns=FEATURE_COLUMNS,
            cost_bps=args.cost_bps,
            label_threshold=args.label_threshold,
        )

        if len(ticker_sequences) == 0:
            skipped_tickers.append(f"{ticker}: no valid trade sequences after preprocessing")
            print(f"Skipping {ticker}: no valid trade sequences after preprocessing.")
            continue

        all_sequences.append(ticker_sequences)
        all_labels.append(ticker_labels)
        all_metadata.append(ticker_metadata)
        print(
            f"Prepared {ticker}: {len(ticker_sequences)} samples, "
            f"win rate={ticker_labels.mean():.3f}"
        )

    if not all_sequences:
        raise ValueError("No training samples were generated. Check the input data and preprocessing settings.")

    full_sequences = np.concatenate(all_sequences, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)
    order = metadata.sort_values(["Signal_Time", "Ticker", "Entry_Time"], kind="stable").index.to_numpy()
    full_sequences = full_sequences[order]
    full_labels = full_labels[order]
    metadata = metadata.loc[order].reset_index(drop=True)
    metadata.insert(0, "sample_id", np.arange(len(metadata), dtype=int))

    split_masks = split_by_signal_time(metadata)
    train_mask = split_masks["train"]
    validation_mask = split_masks["validation"]
    test_mask = split_masks["test"]

    train_split = SplitDataset(
        full_sequences[train_mask],
        full_labels[train_mask],
        metadata.loc[train_mask].reset_index(drop=True),
    )
    validation_split = SplitDataset(
        full_sequences[validation_mask],
        full_labels[validation_mask],
        metadata.loc[validation_mask].reset_index(drop=True),
    )
    test_split = SplitDataset(
        full_sequences[test_mask],
        full_labels[test_mask],
        metadata.loc[test_mask].reset_index(drop=True),
    )

    split_info = {
        "train_start": timestamp_to_iso(split_masks["train_start"]),
        "train_end": timestamp_to_iso(split_masks["train_end"]),
        "validation_start": timestamp_to_iso(split_masks["validation_start"]),
        "validation_end": timestamp_to_iso(split_masks["validation_end"]),
        "test_start": timestamp_to_iso(split_masks["test_start"]),
        "test_end": timestamp_to_iso(split_masks["test_end"]),
    }

    validate_split_order(train_split.metadata, validation_split.metadata, test_split.metadata)
    validate_class_diversity(train_split.labels, "train")
    validate_class_diversity(validation_split.labels, "validation")
    validate_class_diversity(test_split.labels, "test")

    return DatasetBundle(
        full_sequences=full_sequences,
        full_labels=full_labels,
        full_metadata=metadata,
        train=train_split,
        validation=validation_split,
        test=test_split,
        split_info=split_info,
        skipped_tickers=skipped_tickers,
    )


def validate_split_order(train_meta: pd.DataFrame, validation_meta: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    if train_meta["Signal_Time"].max() >= validation_meta["Signal_Time"].min():
        raise ValueError("Train and validation split ordering is invalid.")
    if validation_meta["Signal_Time"].max() >= test_meta["Signal_Time"].min():
        raise ValueError("Validation and test split ordering is invalid.")


def validate_class_diversity(labels: np.ndarray, split_name: str) -> None:
    unique = np.unique(labels.astype(int))
    if len(unique) < 2:
        raise ValueError(f"{split_name.title()} split must contain both classes, got {unique.tolist()}.")


def fit_scaler(train_sequences: np.ndarray) -> RobustScaler:
    _, _, num_features = train_sequences.shape
    scaler = RobustScaler()
    scaler.fit(train_sequences.reshape(-1, num_features))
    return scaler


def transform_sequences(sequences: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    num_samples, lookback, num_features = sequences.shape
    transformed = scaler.transform(sequences.reshape(-1, num_features)).reshape(num_samples, lookback, num_features)
    transformed = transformed.astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Scaled sequences contain NaN or infinite values.")
    return transformed


def make_loader(sequences: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict_logits(
    model: nn.Module,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        torch.tensor(sequences, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_logits = model(batch).cpu().numpy()
            logits.append(batch_logits)

    return np.concatenate(logits, axis=0)


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)

    metrics = {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, clipped)),
        "accuracy": float(accuracy_score(labels, clipped >= 0.5)),
        "mean_predicted_probability": float(clipped.mean()),
        "realized_win_rate": float(labels.mean()),
    }

    if np.unique(labels).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, clipped))
    else:
        metrics["roc_auc"] = None

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_sequences: np.ndarray,
    validation_labels: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[list[dict], int, nn.Module]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            batch_losses.append(loss.item())

        validation_logits = predict_logits(model, validation_sequences, batch_size=batch_size, device=device)
        validation_probs = sigmoid_np(validation_logits)
        validation_loss = float(
            log_loss(validation_labels.astype(int), np.clip(validation_probs, EPS, 1.0 - EPS), labels=[0, 1])
        )
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_log_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"validation_log_loss={validation_loss:.5f}"
        )

        if validation_loss + 1e-8 < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= DEFAULT_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    model.load_state_dict(best_state)
    return history, best_epoch, model


def fit_platt_scaler(logits: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    labels = labels.astype(int)
    if len(logits) == 0 or np.unique(labels).size < 2:
        return {
            "fitted": False,
            "coef": 1.0,
            "intercept": 0.0,
            "reason": "validation split lacks both classes",
        }

    calibrator = LogisticRegression(random_state=seed)
    calibrator.fit(logits.reshape(-1, 1), labels)
    return {
        "fitted": True,
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "reason": None,
    }


def apply_platt_scaler(logits: np.ndarray, calibration: dict) -> np.ndarray:
    return sigmoid_np(calibration["coef"] * logits + calibration["intercept"])


def build_metrics_report(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    history: list[dict],
    best_epoch: int,
    device: torch.device,
    raw_probabilities: dict[str, np.ndarray],
    calibrated_probabilities: dict[str, np.ndarray],
) -> dict:
    return {
        "config": {
            "tickers": args.tickers,
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "lookback": args.lookback,
            "label_threshold": args.label_threshold,
            "cost_bps": args.cost_bps,
            "rsi_os": args.rsi_os,
            "rsi_ob": args.rsi_ob,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "seed": args.seed,
            "feature_columns": FEATURE_COLUMNS,
        },
        "device": str(device),
        "skipped_tickers": bundle.skipped_tickers,
        "dataset": {
            "total_samples": int(len(bundle.full_labels)),
            "train_samples": int(len(bundle.train.labels)),
            "validation_samples": int(len(bundle.validation.labels)),
            "test_samples": int(len(bundle.test.labels)),
            "split_info": bundle.split_info,
        },
        "best_epoch": int(best_epoch),
        "history": history,
        "train": {
            "raw": compute_binary_metrics(bundle.train.labels, raw_probabilities["train"]),
            "calibrated": compute_binary_metrics(bundle.train.labels, calibrated_probabilities["train"]),
        },
        "validation": {
            "raw": compute_binary_metrics(bundle.validation.labels, raw_probabilities["validation"]),
            "calibrated": compute_binary_metrics(bundle.validation.labels, calibrated_probabilities["validation"]),
        },
        "test": {
            "raw": compute_binary_metrics(bundle.test.labels, raw_probabilities["test"]),
            "calibrated": compute_binary_metrics(bundle.test.labels, calibrated_probabilities["test"]),
        },
    }


def save_artifacts(
    args: argparse.Namespace,
    model: nn.Module,
    scaler: RobustScaler,
    calibration: dict,
    metrics: dict,
    bundle: DatasetBundle,
    scaled_splits: dict[str, np.ndarray],
    raw_logits: dict[str, np.ndarray],
    raw_probabilities: dict[str, np.ndarray],
    calibrated_probabilities: dict[str, np.ndarray],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "feature_columns": FEATURE_COLUMNS,
        "lookback": args.lookback,
        "split_info": bundle.split_info,
        "calibration": calibration,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    joblib.dump(scaler, args.output_dir / "scaler.joblib")

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    test_predictions = bundle.test.metadata.copy()
    test_predictions["raw_logit"] = raw_logits["test"]
    test_predictions["raw_probability"] = raw_probabilities["test"]
    test_predictions["calibrated_probability"] = calibrated_probabilities["test"]
    test_predictions["predicted_label"] = (test_predictions["calibrated_probability"] >= 0.5).astype(int)
    test_predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)

    verify_saved_checkpoint(
        model_path=args.output_dir / "model.pt",
        scaler_path=args.output_dir / "scaler.joblib",
        expected_sequences=scaled_splits["test"],
        expected_probabilities=calibrated_probabilities["test"],
        batch_size=args.batch_size,
    )


def verify_saved_checkpoint(
    model_path: Path,
    scaler_path: Path,
    expected_sequences: np.ndarray,
    expected_probabilities: np.ndarray,
    batch_size: int,
) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    reloaded_model = LSTMWinPredictor(**checkpoint["model_config"])
    reloaded_model.load_state_dict(checkpoint["state_dict"])
    reloaded_model.eval()
    _ = joblib.load(scaler_path)

    reloaded_logits = predict_logits(
        reloaded_model,
        expected_sequences,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    reloaded_probabilities = apply_platt_scaler(reloaded_logits, checkpoint["calibration"])

    if not np.allclose(reloaded_probabilities, expected_probabilities, atol=1e-6):
        raise ValueError("Reloaded checkpoint probabilities do not match saved test probabilities.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.lookback < 2:
        raise ValueError("--lookback must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")

    print("Building labeled sequence dataset...")
    bundle = build_dataset_bundle(args)

    scaler = fit_scaler(bundle.train.sequences)
    scaled_splits = {
        "train": transform_sequences(bundle.train.sequences, scaler),
        "validation": transform_sequences(bundle.validation.sequences, scaler),
        "test": transform_sequences(bundle.test.sequences, scaler),
    }

    print(
        f"Dataset summary | train={len(bundle.train.labels)} | "
        f"validation={len(bundle.validation.labels)} | test={len(bundle.test.labels)}"
    )
    print(
        f"Signal date splits | train_end={bundle.split_info['train_end']} | "
        f"validation_end={bundle.split_info['validation_end']} | test_end={bundle.split_info['test_end']}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWinPredictor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(scaled_splits["train"], bundle.train.labels, args.batch_size, shuffle=True)
    history, best_epoch, model = train_model(
        model=model,
        train_loader=train_loader,
        validation_sequences=scaled_splits["validation"],
        validation_labels=bundle.validation.labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    raw_logits = {
        "train": predict_logits(model, scaled_splits["train"], args.batch_size, device),
        "validation": predict_logits(model, scaled_splits["validation"], args.batch_size, device),
        "test": predict_logits(model, scaled_splits["test"], args.batch_size, device),
    }
    raw_probabilities = {name: sigmoid_np(logits) for name, logits in raw_logits.items()}

    calibration = fit_platt_scaler(raw_logits["validation"], bundle.validation.labels, seed=args.seed)
    calibrated_probabilities = {
        name: apply_platt_scaler(logits, calibration) for name, logits in raw_logits.items()
    }

    metrics = build_metrics_report(
        args=args,
        bundle=bundle,
        history=history,
        best_epoch=best_epoch,
        device=device,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
    )

    save_artifacts(
        args=args,
        model=model,
        scaler=scaler,
        calibration=calibration,
        metrics=metrics,
        bundle=bundle,
        scaled_splits=scaled_splits,
        raw_logits=raw_logits,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
    )

    roc_auc = metrics["test"]["calibrated"]["roc_auc"]
    roc_auc_text = "nan" if roc_auc is None else f"{roc_auc:.5f}"
    print(f"Saved artifacts to {args.output_dir}")
    print(
        f"Test calibrated metrics | log_loss={metrics['test']['calibrated']['log_loss']:.5f} | "
        f"roc_auc={roc_auc_text} | "
        f"accuracy={metrics['test']['calibrated']['accuracy']:.5f}"
    )


if __name__ == "__main__":
    main()
