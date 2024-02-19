# **Penguin Classifier**

Made a penguin classifier using CNN with pytorch. The aim was to try making a fun classifier app.

Deployed to streamlit: https://penguinclassifier.streamlit.app/

**Other details of code:**
- Classes of penguins for this classifier: 'Chinstrap Penguins', 'Piplup', 'Adelie Penguins', 'Gentoo Penguins', 'Miniso Penguins' (real species of penguins and animated penguins)
- Data augmentation and normalization
- CNN includes: maxpool2d, batchnorm2d, dropout 0.5

**Other thoughts:**
- The model's accuracy is somewhat low but I will work on the accuracy; the main focus was on the process of trying to create a CNN.
- Original .pth model was too big... so I had to quantize the model.
- Future improvements: Use a pre-trained model, use a model with more layers, deploy an inference endpoint
