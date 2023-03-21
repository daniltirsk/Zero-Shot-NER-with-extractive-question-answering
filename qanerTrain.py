from deeppavlov import train_model
from deeppavlov.core.commands.utils import parse_config

model_config = parse_config('qa_squad2_bert')
model_config['dataset_reader']['data_path'] = './'
model_config['train']['epochs'] = 3
model_config['validation_patience'] = 10
model = train_model(model_config, install=True, download=True)
