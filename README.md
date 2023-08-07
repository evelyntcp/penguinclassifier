# Penguin Classifier
Made a penguin classifier using CNN with pytorch.
The aim was to try out making a fun classifier app.

## Image data acquisition
Google images of chinstrap, adelie, gentoo penguins, as well as fictional penguins like piplup and miniso plush penguin.

## Deployed to streamlit
https://penguinclassifier.streamlit.app/

## Other details of code
- Data augmentation and normalization
- CNN includes: maxpool2d, batchnorm2d, dropout 0.5

## Other thoughts
- The model accuracy is kind of low but I will work on the accuracy, main focus is on the process.
- Also experimented with google colab and having the data in google drive folders.
- Original .pth model was too big (200MB+) for Github... so I had to quantize the model
