## Objective: Understanding LSTM Networks

RNN vs. LSTM:

- RNNs use output of the network to train future iterations of the network
- However, RNNs suffer from the vanishing gradient problem
- RNNs have a hard time storing long term memory
- LSTMs have both long term and short term memory

-------

- Architecture of LSTMs contain the following gates:
    - Learn Gate (short term memory + event) --> remember gate and use gate
    - Forget Gate (long term memory) --> remember gate and use gate 
    - Remember Gate (long term memory + learn gate) --> new long term memory
    - Use Gate (long term memory + learn gate) --> new short term memory (output)

**Learn Gate**

Learn Gate takes STM<sub>t-1</sub> and E<sub>t</sub> as input and runs them through two functions:
- Tanh (combine)
- Sigmoid (ignore)

Output of the Learn Gate is N<sub>t</sub>i<sub>t</sub> where:
- \[
N<sub>t</sub> = tanh(W<sub>n</sub>[STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>n</sub>)
\]
- \[
i<sub>t</sub> = &sigma;(W<sub>i</sub>[STM<sub>t-1</sub>,E<sub>t</sub>]+b<sub>i</sub>)
\]

**Forget Gate**

It takes the long term memory and decides what pieces to keep vs. forget.

Forget Gate takes LTM<sub>t-1</sub> and gets multiplied by a forget factor f<sub>t</sub>.


f<sub>t</sub> gets calculated by taking STM<sub>t-1</sub> and E<sub>t</sub> as inputs and runs them through one function:
- Sigmoid

Output of the Forget Gate is LTM<sub>t-1</sub>f<sub>t</sub> where:
- \[
f<sub>t</sub> = &sigma;(W<sub>f</sub>[STM<sub>t-1</sub>,E<sub>t</sub>]+b<sub>f</sub>)
\]

**Remember Gate**

It's the simplest of all. It takes the long term memory from the Forget Gate and the short term memory from the Learn Gate and combines them together.

It simply adds the output from the Forget Gate and the output from the Learn Gate.

Output of the Remember Gate is:
- \[
LTM<sub>t</sub> = LTM<sub>t-1</sub>f<sub>t</sub> + N<sub>t</sub>i<sub>t</sub>\]

**Use Gate**

Use Gate is the output gate. It takes the output of the Forget Gate and the Learn Gate to come up with the New Short Term Memory.

Mathematically, it applies a neural network with a tanh activation on the output of the Forget Gate (U<sub>t</sub>) and another neural network with a Sigmoid activation (V<sub>t</sub>) on the STM<sub>t-1</sub> and E<sub>t</sub> and finally multiplies them together to get to the New Short Term Memory.

Output of the Use Gate is STM<sub>t</sub> = U<sub>t</sub>V<sub>t</sub>, where:
- \[
U<sub>t</sub> = tanh(W<sub>u</sub>LTM<sub>t-1</sub>f<sub>t</sub> + b<sub>u</sub>
    \]
- \[
V<sub>t</sub> = &sigma;(W<sub>v</sub>[STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>v</sub>)
\]

------

**Notes:**
- It is important to remember that the specific function and mathematical operations shown in the above architecture are arbitrary. They are chosen because they have been shown to work.
- However, this is an area of active research and new architectures should be explored further.
- Gated Recurrent Unit (GRU) is an example of an alternative architecture that takes working memory and the event as input --> runs it through an Update Gate followed by a Combine Gate --> to output the New Working Memory.
    - The Combine Gate takes the output of the Update Gate and the original Working Memory as inputs
- Another example of an alternative architecture is a LSTM with Peephole connections
    - Here, you include LTM<sub>t-1</sub> as another input to the forget function in the Forget Gate (i.e., the long term memory now has an influence on what is forgotten)
    - Mathematically, the new forget function is as follows: f<sub>t</sub> = &sigma;(W<sub>f</sub>[LTM<sub>t-1</sub>, <sub>t-1</sub>, E<sub>t</sub>]+b<sub>f</sub>)
    - In a LSTM with Peephole connections, you introduce this connection to every sigmoid function in the LSTM cell
