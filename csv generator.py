import os
import pandas as pd

for dataset in os.listdir('./datasets'):
    for d in os.listdir(f'./datasets/{dataset}'):
        df = pd.DataFrame(columns=['path', 'label'])
        for part in os.listdir(f'./datasets/{dataset}/{d}'):
            for img in os.listdir(f'./datasets/{dataset}/{d}/{part}'):
                df.loc[len(df)] = {'path': f'./datasets/{dataset}/{d}/{part}/{img}', 'label': part[:-1]}

        df.to_csv(f'./ludwig-data/{d}.csv', index=False)
