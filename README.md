# Psychic GAN - A self-attentive GAN for future video frame generation
Andreas Pirchner / Aleksandar Levic
## Architecture
Our architecture consists of three major components:
 - The `encoder` takes `n` video frames as input and encodes them as vector in a 128-dim. latent space
 - The `generator` generates `m` video frames conditional on the encoded input
 - The `discriminator` takes both the generated frames and the corresponding ground truth frames and tries to classify them into real and fake
![arch](https://user-images.githubusercontent.com/56101896/69471531-3667e980-0ddb-11ea-8da6-8f85cf526c76.png)
