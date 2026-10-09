from src.model import train_models

if __name__ == '__main__':
    metrics=train_models('data/personalised_dataset.csv')
    print('Training complete')
    for name, m in metrics.items():
        print(f'{name}: accuracy={m["accuracy"]}, weighted_f1={m["weighted_f1"]}')
