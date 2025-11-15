# Pokemon Type Classifier Project
This archive contains the majority of the work for our type classifier project. This README serves as a guide to the files listed here.

# Folders
## archive
This folder contains the origninal images, additional images, and the csv files we gathered and generated that made up our dataset for this project.
## archived checkpoints
This folder contains an archive for various checkpoints for models that we had trained, for all 3 attempted models.
## checkpoints
This folder is the current checkpoitns that are overwritten when new runs are done.
## lightning_logs
A logging folder for the lightning files generated during our AI training.
## pokemon_type_gradio
A backup of the demo code found on Hugging Face, minor some final tweaks that made the demo fully up to date.
## pokemon_type_gradio_efficientNet
A efficient net version of our Hugging Face demo, when we were concidering doing multiple models for the demo

# Loose Files
## pokemon_features_with_labels
The image archive generated from our orignal transfer learning process.
## pokemonDataAnalysis
Our notebook containing our original data analysis and regressor, that provided our baselines using the original dataset.
## pokemonDataAnalysisAdditional
Our notebook containing our second data analysis and regressor, that provided our baselines for the updated dataset, that we used as our actual baseline.
## pokemonDatasetProcessing
Our notebook containing our work to load the second dataset that we had found, which is used in any of the DualType files.
## pokemonDuelTypeEfficientNet
Our complete network running with EfficientNet transfer learning.
## pokemonDuelTypeResNet
Our complete network running with ResNet transfer learning.
## pokemonDuelTypeWithWeight
Our complete network that was used for the demo
## pokemonSingleType
Our single type network that we had constructed
## pokemonSingleTypeImageLoading
Our single type image loader that was custom written for this project.
## pokemonSingleTypeWithWeight
Our improvement to the orginal single type network that relied on image transformation and weighting the training by the number of class examples that we had.
## pokemonSingleTypeProcessing
A helper notebook written to help us with the updated image file and original tabular data