# Minimal MNIST

[Second Demo run of Minimal MNIST (after fixing mistake discussed in notes)](docs/demo.png)

Sunday challenge to solve MNIST as fast as possible in rust with no external dependencies (save rng and plotting.) Not a very fast implementation (only had time really to write, debug and run), but it gets fairly accurate. Since I had so much fun this will be an ongoing recreational programming project.

Next steps will be to make this faster (and more efficient, lots of memory waste currently), and multithreaded as well (thought this might require some other additions first). After that I'll start testing out ReLU, adding other kinds of random initialization, weight saving (coming first actually!), and perhaps a refactor to help out various tests (more hidden layers, and all the changes discussed). In the future it would be cool to build a simple autograd engine, our own tensor class and eventually the beginnings of a computational graph engine (to test out other kinds of architectures vs MNIST, particularly some conv blocks, dropout, and gelu)

## Architecture

Current implemented architecture is your bog standard MLP. 784 -> 512 -> 10 neurons per layer, with sigmoid as the activation function for the hidden layer and of course softmax on the output. Cross-entropy loss, SGD (without momentum), standard uniform distribution weight initialization (its what rand::random_range provides).

## Metrics

Not very involved, just a look at accuracy, loss over time and a confusion matrix.

## Notes

First attempt failed because random initialization of weights was in fact used, however I didn't consider the fact that there would need negative values rather than the range 0-1(silly me). Naturally this would immediately disallow future training because were using sigmoid, which will saturate to 1 with these values, and the calculated gradient will stay at 0.

For now well initialize between -1 and 1, and potentially switch to Xavier initialization later (this was definitely not neccessary).
