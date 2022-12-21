import tensorflow as tf

def score(data,model, verbose):
    score = model.evaluate(data, verbose=verbose)
    print("Loss:", score[0], "\n", "Accuracy:", score[1])