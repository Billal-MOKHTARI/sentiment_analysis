# sentiment_analysis

![1__JW1JaMpK_fVGld8pd1_JQ](https://user-images.githubusercontent.com/100322125/177759134-de507efb-a9ba-44a1-98e1-568e8d260994.gif)

Given some review, the algorithm tells us if it is good or bad, by using different Neural Network architectures including <strong>GRU</strong>, <strong>LSTM</strong>, <strong>1D Convolutional layers.</strong>


<h4>Requirements :</h4>
<strong>Python version 3.8</strong> because the lattest version doesn't support <strong>tensorflow_datasets</strong>

## Embedding projector

![ezgif com-gif-maker](https://user-images.githubusercontent.com/100322125/178844316-827567c7-4233-4563-9bff-9086917b902b.gif)

The embedding vectors tell us about correlation between the words. The metric used to decide if the words belong to the same class, is called <strong>cosine similarity</strong>. If the vectors <strong>A</strong> and <strong>B</strong> have almost the same orientaton, $$Cosine Similarity(A, B) \approx 0$$


$$Cosine Similarity(A, B)=cos(\theta)=\frac{A \cdot B}{|A|\cdot|B|}=\frac{|A|\cdot|B|\cdot cos(\theta)}{|A|\cdot|B|}=\frac{A^{T}B}{\sqrt{A^{T}}\sqrt{B^{T}}}$$
