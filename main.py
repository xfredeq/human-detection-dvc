from ludwig.api import LudwigModel
import pandas as pd

train_df = pd.read_csv('ludwig-data/horse-or-human.csv')
print(train_df.head())

model_definition = {
    'input_features': [
        {
            'name': 'path', 'type': 'image',
            'preprocessing': {
                'resize_method': 'crop_or_pad', 'width': 64,
                'height': 64
            }
        }],
    'output_features': [{'name': 'label', 'type': 'category'}]
}

model = LudwigModel(model_definition, logging_level=25)
train_stats, _, _ = model.train(dataset=train_df)

print(train_stats)

test_df = pd.read_csv('ludwig-data/validation-horse-or-human.csv')
print(test_df.head())

model.predict(dataset=test_df)
