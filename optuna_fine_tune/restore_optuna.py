import optuna
study = optuna.create_study(study_name='training-DM023', storage='sqlite:///example.db', load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
# print(df)

optuna.visualization.plot_contour(study, params=['Learning_rate', 'NUM_EPOCHS'])