# Mobility Model Card for "add model name here"

*For explanations of the sections and example model cards, please revisit the orginal [Google paper](https://arxiv.org/abs/1810.03993) and our [Mobility Model Card manuscript](EMERALDS_Mobility_Model_Cards_submitted.pdf)*
  
Jump to section:

- [Model details](#model-details)
- [Intended use](#intended-use)
- [Factors](#factors)
- [Metrics](#metrics)
- [Evaluation data](#evaluation-data)
- [Training data](#training-data)
- [Quantitative analyses](#quantitative-analyses)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Model details

### Person or organization developing model

*What person or organization developed the model? This can be used by all stakeholders to infer details pertaining to model development and potential conflicts of interest.*

### Model date

*When was the model developed? This is useful for all stakeholders to become further informed on what techniques and data sources were likely to be available during model development*

### Model version

*Which version of the model is it, and how does it differ from previous versions? This is useful for all stakeholders to track whether the model is the latest version, associate known bugs to the correct model versions, and aid in model comparisons.*

### Unique identifier. 

*In addition to the model version, a unique identifier should be assigned to facilitate tracking and linking between related models. This could be implemented using universally unique identifiers (UUIDs) or other standardized referencing systems. (See also the “Related model cards” addition to the Caveats and Recommendations section.)*

### Model type

*What type of model is it? This includes basic model architecture details, such as whether it is a Naive Bayes classifier, a Convolutional Neural Network, etc. This is likely to be particularly relevant for software and model developers, as well as individuals knowledgeable about machine learning, to highlight what kinds of assumptions are encoded in the system.*

### Geographic area covered

*Mobility models are often location specific. A model trained in one city may not be directly transferable to another. Therefore, specifying the geographic coverage—whether as named regions, a bounding box, or a map plot—is crucial for identifying suitable models. Additionally, if the model has been trained to provide predictions for a certain time only, its valid time extent should also be clearly stated*

###  Information about training algorithms, parameters, fairness constraints or other applied  approaches, and features

### Paper or other resource for more information

*Where can resources for more information be found?*

### Citation details

*How should the model be cited?*

### License

*License information can be provided.*

### Contact details 

*Where to send questions or comments about the model.*

## Intended use

### Primary intended uses

*Mobility models serve diverse use cases, including trajectory prediction and imputation, travel or arrival time estimation, (sub)trajectory classification, anomaly detection, next-location and destination prediction, synthetic data generation, location classification, and traffic volume or crowd flow prediction. Clearly specifying the model’s intended application helps ensure appropriate usage.*

### Prediction horizon

*Since many mobility models involve forecasting, it is essential to indicate the model’s prediction horizon, i.e., how far into the future predictions can be generated, if applicable. This helps users assess the model’s suitability for specific planning or real-time applications.*

### Example results

*Providing sample outputs, such as predictions or classifications, can illustrate the model’s behavior and expected performance in practical scenarios.*

### Primary intended users

*Some mobility models are robust enough for real-time traffic control, while others are designed for urban planning, transport policy analysis, or research. Identifying the primary audience helps align expectations regarding model capabilities. This may also include whether the model expects data in a streaming format or batch processing.*

### Transferability

*Mobility models often depend on specific geographic regions, transport networks, or modes of transport. For example, a travel time prediction model trained on taxi data may not generalize well to private vehicles, as taxis often have access to dedicated lanes. It is important to specify whether the model can be applied to other regions, network sections, or transport modes without retraining. If retraining is required, this should be explicitly noted.*

### Out-of-scope use cases

*Certain mobility scenarios, such as extreme events (e.g., road closures, accidents, or natural disasters), may be underrepresented or entirely missing from the training data. The Model Card should specify these limitations to avoid misleading interpretations of model outputs.*

## Factors

### Mobility context

*Mobility models may perform differently depending on their application context. Factors such as road type (e.g., highways vs. urban streets), mode of transport (e.g., motorized vehicles vs. walking or cycling), user groups (e.g., wheelchair users, individuals without private vehicles), or geographic setting (e.g., urban vs. rural areas) should be explicitly documented.*

### Environmental conditions

*Model performance can vary significantly based on environmental factors. For example, a model may perform well under normal weather conditions but worse during bad or extreme weather events. Clearly stating the environmental conditions under which the model can provide reliable results can help users anticipate potential limitations.*

### Evaluation factors

*Which factors are being reported, and why were these chosen? If the relevant factors and evaluation factors are different, why?*

## Metrics

### Model performance measures

*The choice of evaluation metrics depends on the specific mobility use case. Commonly used metrics include Mean Absolute Percentage Error (MAPE), Coefficient of Determination (R2), Root Mean Squared Error (RMSE), and Precision and Recall for classification-based mobility tasks. A structured overview of relevant metrics is provided in [7].*

### Decision thresholds

*To contextualize performance measurements, it is important to specify the thresholds used in the evaluation. These may include the forecast horizon (e.g., 15-minute or 1-hour predictions), spatial resolution (e.g., traffic density per road segment), and temporal resolution (e.g., hourly or daily mobility forecasts).*

### Approaches to uncertainty and variability: 

*How are the measurements and estimations of these metrics calculated? For example, this may include standard deviation, variance, confidence intervals, or KL divergence. Details of how these values are approximated should also be included (e.g., average of 5 runs, 10-fold cross-validation).*



## Training data

### Datasets

*Clearly describing the spatial and temporal coverage of the training and evaluation data is crucial. This includes details such as the geographic extent, time span, and sampling interval of mobility time series. If publicly available datasets were used, persistent identifiers (e.g., Zenodo IDs) should be provided for reproducibility.*

### Motivation

*Why were these datasets chosen?*

### Preprocessing

*Mobility data often contains quality issues such as missing values, outliers, or inconsistencies that must be addressed before training. The Model Card should describe preprocessing steps, including data cleaning methods, resampling strategies,
discretization, and aggregation processes. Transparency in these procedures ensures a better understanding of model reliability and potential limitations.*

## Evaluation data

### Datasets

*Clearly specifying the spatial and temporal coverage of evaluation data helps users understand the conditions under which the model was tested. Information should include the geographic areas ane time spans.*

### Motivation

*Why were these datasets chosen?*

### Preprocessing

*Proper handling of spatial and temporal correlations in the evaluation dataset is crucial. The Model Card should describe the methodology used to split training and evaluation data, ensuring that no data leakage occurs. In mobility predictions, time series-based splits are often required to accurately reflect real-world deployment conditions.*

## Quantitative analyses

### Unitary results

*How did the model perform with respect to each factor?*

### Intersectional result

*How did the model perform with respect to the intersection of evaluated factors?*

### Multidimensionality 

*Mobility models often generate outputs across multiple dimensions, such as demographic, geographic, and temporal factors. Evaluating model performance separately for each of these dimensions helps identify biases and assess whether the model generalizes well across different user groups and environments.*

### Analysis units

*Model results should be assessed at the most granular meaningful level. For example, if a model predicts ride demand at the station level but is used for area-wide forecasts (aggregating multiple stations), performance should be evaluated both at the station level and in aggregate.*

## Ethical considerations

### Environmental impact

*The energy and resource consumption of training and deploying mobility models should be documented. This includes details on the location of cloud computing centers, their energy and water consumption, and the carbon footprint of the computational infrastructure used [12]. Transparency in these factors supports sustainable AI development.*

### Societal impact

*Mobility models influence transportation planning and urban policies, potentially affecting different populations unequally. It is important to assess who benefits from the model and whether any communities may be disadvantaged. Additionally, the Model Card should document the potential impact on workers involved in labeling the training data, including where these workers are located and under what conditions they performed their tasks. This ensures that ethical considerations extend beyond model deployment to the full lifecycle of AI development.*

### Biases

*Mobility models may exhibit geographic, demographic, or temporal biases based on the data used for training. These biases should be explicitly evaluated and reported in Mobility Model Cards, along with any strategies used to mitigate them.*

### Explainability

*Model transparency is a growing concern for stakeholders. Mobility Model Cards should document whether the model is inherently interpretable or if post-hoc explainability techniques, such as SHAP, LIME, or permutation importance, have been applied to analyze model predictions.*

## Caveats and recommendations

### Recent legal / environmental changes

*Mobility models may not account for recent regulatory changes, such as new speed limits, toll policies, or driving restrictions. Similarly, environmental factors, such as infrastructure modifications or public transit expansions, can affect model predictions. Users should be aware that real-world changes may reduce model accuracy over time.*

### Technical dependencies

*Mobility models require specific software libraries or packages to function correctly. A list of dependencies should be provided, ideally as a requirements file, to ensure reproducibility and ease of deployment.*

### Retraining

*Mobility models often experience performance degradation due to data drift, shifts in mobility patterns, or changes in spatial networks. The Model Card should specify whether and how frequently retraining is required to maintain accuracy.*

### Related model cards

*Listing related Model Cards can help users understand connections between models and identify alternative options. (Some catalogs, such as Huggingface, have already adopted this approach.) Additionally, leveraging semantic web approaches for linking Model Cards as structured data has been proposed [6], which could further improve trust and reusability.*
