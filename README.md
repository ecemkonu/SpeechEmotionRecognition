# Speech Emotion Recognition(SER)

Keras implementation for Speech emotion recognition using deep 1D & 2D CNN LSTM networks by Zhoa, Mao and Chen[1]. 1dcnn LSTM network is heavily
modified to perform on a public dataset, The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)[2]. 1dcnn LSTM architecture takes speech signal as input, meanwhile 2dcnn lstm network will use mel spectogram features, aka representation of speech signal in frequency domain.
However, it differs from fourier transform as it is scaled to hearing range of humans.

[1]https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337


[2]https://zenodo.org/record/1188976
