# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import os
import pickle
import logging
import threading
import time
from prometheus_client import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = 'models/fraud_model.pkl'
CANDIDATE_PATH   = 'models/fraud_model_candidate.pkl'
FEATURES_PATH    = 'models/feature_names.pkl'

# ── Prometheus counter ────────────────────────────────────────────────────────
MODEL_SWAPS = Counter(
    'fraud_api_model_swaps_total',
    'Total number of zero-downtime model swaps performed'
)

class ModelWatcher:
    """
    Watches for a candidate model file and swaps it into the API
    without restarting. Runs as a background thread.
    """

    def __init__(self, app_state, check_interval=30):
        """
        app_state: a dict shared with main.py containing
                   'model', 'feature_names', 'model_loaded'
        check_interval: seconds between checks
        """
        self.app_state     = app_state
        self.check_interval = check_interval
        self._thread       = None
        self._running      = False

    def _load_candidate(self):
        """Load the candidate model and swap it in."""
        try:
            with open(CANDIDATE_PATH, 'rb') as f:
                new_model = pickle.load(f)
            with open(FEATURES_PATH, 'rb') as f:
                new_features = pickle.load(f)

            # Atomic swap -- update shared state
            self.app_state['model']         = new_model
            self.app_state['feature_names'] = new_features
            self.app_state['model_loaded']  = True

            # Replace current model with candidate
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(new_model, f)

            # Remove candidate file so we don't swap again
            os.remove(CANDIDATE_PATH)

            # Increment Prometheus counter
            MODEL_SWAPS.inc()

            logger.info(
                "Zero-downtime model swap complete. "
                "New model is now serving predictions."
            )
            return True

        except Exception as e:
            logger.error(f"Model swap failed: {e}")
            return False

    def _watch_loop(self):
        """Background loop that checks for candidate model every N seconds."""
        logger.info(
            f"Model watcher started — "
            f"checking every {self.check_interval}s"
        )
        while self._running:
            if os.path.exists(CANDIDATE_PATH):
                logger.info("Candidate model detected — initiating swap...")
                self._load_candidate()
            time.sleep(self.check_interval)

    def start(self):
        """Start the background watcher thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._watch_loop,
            daemon=True,    # dies when main process exits
            name='ModelWatcher'
        )
        self._thread.start()
        logger.info("Model watcher thread started")

    def stop(self):
        """Stop the background watcher thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Model watcher stopped")