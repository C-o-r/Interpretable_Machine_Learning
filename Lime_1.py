# First, you'll need to install lime if you haven't already
# pip install lime

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups

# Load dataset
categories = ['alt.atheism', 'soc.religion.christian']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# Create a pipeline with a vectorizer and a classifier
pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Train the model
pipeline.fit(train.data, train.target)

# Instantiate LimeTextExplainer
explainer = LimeTextExplainer(class_names=['atheism', 'christian'])

# Choose an instance from the test set to explain
idx = 1
test_instance = test.data[idx]

# Generate explanation for the chosen instance
exp = explainer.explain_instance(test_instance, pipeline.predict_proba, num_features=6)

# Visualize the explanation
print('Document id: %d' % idx)
print('Predicted class:', categories[pipeline.predict([test_instance])[0]])
print('True class:', categories[test.target[idx]])
exp.show_in_notebook(text=True)
