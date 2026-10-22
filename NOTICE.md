# Attribution and license scope

The MIT license in `LICENSE` applies to IntentOps project source code and documentation.
It does not replace licenses or attribution requirements for third-party data or packages.

The frozen baseline was trained using the published BANKING77 data from PolyAI Limited,
made available under Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

Dataset: https://huggingface.co/datasets/PolyAI/banking77
Original data: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

Credit: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić,
“Efficient Intent Detection with Dual Sentence Encoders,” Proceedings of the 2nd Workshop
on NLP for Conversational AI, 2020. https://arxiv.org/abs/2003.04807

Changes made: original CSV files were converted to Parquet with integer labels mapped
to intent names. A stratified development subset fitted a TF-IDF/logistic-regression
pipeline. The official test data was not filtered or relabeled. The deployment bundle
contains the frozen trained artifact and aggregate results, not the original dataset.
Retain this attribution when redistributing the artifact; the code's MIT license does
not purport to relicense the dataset. No affiliation with or endorsement by PolyAI is implied.

Installed dependencies retain their respective licenses. No pretrained semantic encoder
weights are included in the default release.
