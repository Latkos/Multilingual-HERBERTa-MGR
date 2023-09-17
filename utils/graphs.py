import pandas as pd
from transformers import AutoTokenizer, pipeline
import plotly.express as px



def evaluate_model_on_subset(model, model_path, subset, test_df, language_col='lang'):
    filtered_df = test_df[test_df[language_col].isin(subset)]
    evaluation_results = model.evaluate(df=filtered_df, model_path=model_path)
    return evaluation_results


def plot_f1_depending_on_column(model, model_paths, test_df, col='lang', subset=None):
    if subset is None:
        subset = test_df[col].unique()
    results = []
    for model_path in model_paths:
        model_f1 = []
        for element in subset:
            evaluation_results = evaluate_model_on_subset(model, model_path, col, [element], test_df)
            f1 = evaluation_results['eval_overall_f1']
            model_f1.append(f1)
        results.append((model_path, model_f1))
    f1_df = pd.DataFrame(results, columns=['Model', 'F1 Score'])
    f1_df = pd.concat([f1_df, pd.DataFrame(f1_df['F1 Score'].tolist(), columns=subset)], axis=1)
    fig = px.imshow(f1_df.set_index('Model'),
                    labels=dict(x=f"{col}", y="Models", color="F1 Score"),
                    color_continuous_scale='Viridis',
                    zmin=0.0, zmax=1.0)
    fig.update_layout(title="F1 Scores for Models on Different Languages",
                      xaxis_title="Languages",
                      yaxis_title="Models")
    fig.show()
    return fig
