from ml.training.train import train

for pair in ['EUR_USD', 'AUD_USD', 'NZD_USD']:
    print(f'Training {pair}...')
    r = train(pair_name=pair, epochs=50, patience=10, seed=42)
    print(f'{pair} done - best val loss: {r["best_val_loss"]:.6f}')

print('All models trained successfully.')
