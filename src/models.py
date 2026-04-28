class State:
    model = None
    tfidf = None
    ready = False
    baselines = {}
    retraining = False
    last_retrain_time = None
    last_retrain_status = None

