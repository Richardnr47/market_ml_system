def run_train_pipeline(config_path: str) -> dict:
    print("\n==============================")
    print("[PIPELINE] START TRAIN PIPELINE")
    print("==============================\n")

    cfg = load_yaml(config_path)

    print("[CONFIG] Loaded config")

    print("[DATA] Loading dataset...")
    df = pd.read_csv(cfg["dataset"]["input_file"])

    print(f"[DATA] Rows: {len(df)}")
    print(f"[DATA] Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df[cfg["dataset"]["timestamp_col"]])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("[FEATURES] Building market features...")
    df = build_market_features(
        df=df,
        price_col=cfg["dataset"]["price_col"],
        return_lags=cfg["features"]["return_lags"],
        vol_windows=cfg["features"]["vol_windows"],
        momentum_windows=cfg["features"]["momentum_windows"],
        volume_windows=cfg["features"]["volume_windows"],
    )

    print("[TARGET] Building forward return target...")
    df = build_forward_return_target(
        df=df,
        price_col=cfg["dataset"]["price_col"],
        horizon=cfg["dataset"]["target_horizon"],
    )

    feature_cols = [
        c for c in df.columns
        if c.startswith("ret_")
        or c.startswith("rv_")
        or c.startswith("mom_")
        or c.startswith("vol_z_")
        or c in {"hl_range", "oc_change", "body_to_range"}
    ]

    print(f"[FEATURES] Number of features: {len(feature_cols)}")

    print("[WINDOWS] Creating rolling windows...")
    wb = make_windows(
        df=df,
        feature_cols=feature_cols,
        target_col="target_forward_return",
        timestamp_col="timestamp",
        window_size=cfg["windows"]["size"],
        stride=cfg["windows"]["stride"],
    )

    print(f"[WINDOWS] X shape: {wb.X.shape}")
    print(f"[WINDOWS] y shape: {wb.y.shape}")

    X = flatten_windows(wb.X)
    y = wb.y

    print(f"[WINDOWS] Flattened X shape: {X.shape}")

    print("[SPLIT] Generating walk-forward splits...")
    splits = generate_walk_forward_splits(
        n_samples=len(y),
        train_min_size=cfg["validation"]["train_min_size"],
        test_size=cfg["validation"]["test_size"],
        n_splits=cfg["validation"]["n_splits"],
    )

    print(f"[SPLIT] Number of folds: {len(splits)}")

    all_fold_metrics = []

    print("\n[TRAINING] Starting walk-forward training...\n")

    for fold_idx, split in enumerate(splits):
        print(f"------------------------------")
        print(f"[FOLD {fold_idx}]")

        X_train = X[split.train_start:split.train_end]
        y_train = y[split.train_start:split.train_end]

        X_test = X[split.test_start:split.test_end]
        y_test = y[split.test_start:split.test_end]

        print(f"[FOLD {fold_idx}] Train size: {len(X_train)}")
        print(f"[FOLD {fold_idx}] Test size: {len(X_test)}")

        print(f"[FOLD {fold_idx}] Training state model...")
        state_model = GMMStateModel(
            n_components=cfg["state_model"]["n_components"],
            random_state=cfg["state_model"]["random_state"],
        ).fit(X_train)

        print(f"[FOLD {fold_idx}] Transforming state features...")
        X_train_meta = np.column_stack([
            X_train,
            state_model.predict(X_train),
            state_model.predict_proba(X_train),
        ])

        X_test_meta = np.column_stack([
            X_test,
            state_model.predict(X_test),
            state_model.predict_proba(X_test),
        ])

        print(f"[FOLD {fold_idx}] Training predictor...")
        predictor = LGBMReturnPredictor(
            learning_rate=cfg["predictor"]["learning_rate"],
            n_estimators=cfg["predictor"]["n_estimators"],
            num_leaves=cfg["predictor"]["num_leaves"],
            max_depth=cfg["predictor"]["max_depth"],
            subsample=cfg["predictor"]["subsample"],
            colsample_bytree=cfg["predictor"]["colsample_bytree"],
            random_state=cfg["predictor"]["random_state"],
        ).fit(X_train_meta, y_train)

        print(f"[FOLD {fold_idx}] Predicting...")
        preds = predictor.predict(X_test_meta)

        metrics = regression_metrics(y_test, preds)
        print(f"[FOLD {fold_idx}] Metrics: {metrics}")

        all_fold_metrics.append(metrics)

    print("\n[SUMMARY] Calculating average metrics...")
    avg_mae = float(np.mean([m["mae"] for m in all_fold_metrics]))
    avg_r2 = float(np.mean([m["r2"] for m in all_fold_metrics]))

    summary = {
        "avg_mae": avg_mae,
        "avg_r2": avg_r2,
        "n_folds": len(all_fold_metrics),
    }

    print("\n==============================")
    print("[PIPELINE] FINISHED")
    print("==============================")
    print(f"[RESULT] {summary}\n")

    return summary