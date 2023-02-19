from ctcovseg import Config, train

if __name__ == "__main__":
    config = Config()
    metrics = train(config, "input/pngs")
    for metric, value in metrics.items():
        print(metric, value)
