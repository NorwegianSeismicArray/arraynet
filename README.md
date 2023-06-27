<a name="readme-top"></a>

# ArrayNet: A combined seismic phase classification and back-azimuth regression neural network for array processing pipelines, 2023. 

Code and examples related to the paper **ArrayNet: A combined seismic phase classification and back-azimuth regression neural network for array processing pipelines**. 

To train an ArrayNet model (super-model) for the ARCES array run :
```
bash run.sh
```
Edit the script and adjust to your working environment

Input data in `tf/data`:

* `times_merged_arces_4Fre.np`y : time stamp for seismic arrivals
* `X_merged_arces_4Fre.np`y : co-array phase patterns for all arrivals
* `y_cl_merged_arces_4Fre.np`y : Arrival label (phase type)
* `y_reg_merged_arces_4Fre.np`y : Back-azimuth to event source

Due to limitations on file size on github we provide a reduced data set for training. However, we also provide the model trained with the full data set in:

`tf/output_full/`

The model trained with the reduced data set which can be reproduced is in:

`tf/output/`

Call this script to evaluate the model trained with the full data set (Confusion matrix, classification metrics, back-azimuth residuals):
```
python evaluate_models.py
```


Coming soon:

* Scripts to generate input data from raw array waveforms
* Scripts for sub-models


## Related publication

- Andreas Köhler, Erik B. Myklebust. **ArrayNet: A combined seismic phase classification and back-azimuth regression neural network for array processing pipelines**. In revisions for publication in BSSA, 2023.
<!-- ([arXiv](https://arxiv.org/abs/2112.04605)) ([Paper](http://semantic-web-journal.org/content/prediction-adverse-biological-effects-chemicals-using-knowledge-graph-embeddings-0)) ([REPOSITORY](https://github.com/NIVA-Knowledge-Graph/KGs_and_Effect_Prediction_2020)) -->

<a name="readme-top"></a>

## License

See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Andreas Köhler - andreas.kohler@norsar.no - [ORCID](https://orcid.org/0000-0002-1060-7637)

Erik B. Myklebust - [ORCID](https://orcid.org/0000-0002-3056-2544)


Project Link: [https://github.com/NorwegianSeismicArray/arraynet](https://github.com/NorwegianSeismicArray/arraynet)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* ArrayNet models are built with [TensorFLow](https://www.tensorflow.org/)
* ARCES waveform data from which the input data was generated are available via the [Norwegian EIDA node](https://eida.geo.uib.no/webdc3/)
* Reviewed seismic event bulletins from which the input data labels were obtained are available from the [Finish National Seismic Network](https://www.seismo.helsinki.fi/bulletin/list/norBull.html
) and [NORSAR](http://www.norsardata.no/NDC/bulletins/regional/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

