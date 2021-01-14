Colourizing the black and white images using python and tensorflow.
Used pre-trained VGG19 neural network.

Trained the model on WIDER Face dataset that consits of 32K images includes [train, test, val]


Examples of model output
<p>
  <img src="https://github.com/MANOJ-KUMAR-2000/BlackAndWhite-COLOURIZER/blob/main/Colourize_v0/Examples/gray/Example-5.jpg" height="280" width="350" title="Gray image">
  <img src="https://github.com/MANOJ-KUMAR-2000/BlackAndWhite-COLOURIZER/blob/main/Colourize_v0/Examples/coloured/Example-5.png" height="280" width="350" title="Coloured image">
</p>

<p>
  <img src="https://github.com/MANOJ-KUMAR-2000/BlackAndWhite-COLOURIZER/blob/main/Colourize_v0/Examples/gray/Example-4.jfif" height="280" width="350" title="Gray image">
  <img src="https://github.com/MANOJ-KUMAR-2000/BlackAndWhite-COLOURIZER/blob/main/Colourize_v0/Examples/coloured/Example-4.png" height="280" width="350" title="Coloured image">
</p>

The vgg19 model output is concatenated and that is further reduced to 2-channel output the model reference 

<p>
  <img src="https://github.com/MANOJ-KUMAR-2000/BlackAndWhite-COLOURIZER/blob/main/Colourize_v0/model/hypercolumns.png" height="360" width="260" title="Gray image">
</p>

Reference : https://tinyclouds.org/colorize/    

Dataset : http://shuoyang1213.me/WIDERFACE/
