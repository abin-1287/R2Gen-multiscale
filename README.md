# R2Gen-multiscale

### Different from R2Gen

encoder_decoder.py

在multiscale中分别选取hidden——states最后49，196，784维训练decoder

![image-20240517103845692](pic/image-20240517103845692.png)



encoder_decoder2.py

在multiscale中切分49，196，784维训练三个decoder

![image-20240517104411878](pic/image-20240517104411878.png)



VisualExtractor_2.py

add multiscale

![image-20240517104954321](pic/image-20240517104954321.png)
