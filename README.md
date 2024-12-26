# Hawiya: Semantic Search for Arabic Poetry

Hawiya is a project aimed at performing semantic search for Arabic poetry. It leverages advanced techniques in natural language processing (NLP) and machine learning to understand and retrieve relevant poems based on user queries, enhancing the experience of exploring Arabic literature. The project includes various notebooks that explore different stages of the semantic search pipeline, from data cleaning to fine-tuning and implementing multiple models for search and generation.

## Notebooks Overview

Below is a list of the notebooks included in this project:

### 1. `1_datasets_cleaning.ipynb`
This notebook focuses on cleaning and preprocessing the Arabic poetry dataset. It handles removing unwanted characters, standardizing text, and ensuring the dataset is ready for further processing.

### 2. `2_tokenization_embedding.ipynb`
This notebook demonstrates how to tokenize the Arabic text and generate embeddings for words and phrases using pre-trained models, preparing the text for semantic understanding.

### 3. `3_semantic_search.ipynb`
This notebook implements the core semantic search pipeline. It takes a user query and performs a search over the Arabic poetry dataset based on semantic relevance rather than simple keyword matching.

### 4. `4_arabert_tokenization.ipynb`
Focuses on tokenizing Arabic text specifically using AraBERT, a pre-trained model for Arabic. This notebook explores the tokenization process tailored for Arabic poetry.

### 5. `5_arabert_doc2vec.ipynb`
This notebook combines AraBERT with the Doc2Vec algorithm to generate document-level embeddings, allowing us to perform similarity searches over larger chunks of poetry.

### 6. `6_bert_based_search.ipynb`
This notebook implements a BERT-based semantic search method, where we fine-tune BERT to rank Arabic poetry documents based on relevance to a query.

### 7. `7_conditional_generation.ipynb`
Explores conditional text generation using models like GPT to generate poetry or text conditioned on a specific input or prompt.

### 8. `8_fine_tuning.ipynb`
Focuses on fine-tuning a language model (such as BERT or AraBERT) on the specific task of Arabic poetry search, improving the model's ability to understand poetic nuances.

### 9. `9_fine_tuning_v2.ipynb`
An updated version of the fine-tuning notebook with improved techniques, hyperparameter optimization, and performance evaluation.

### 10. `10_sentence_transformer.ipynb`
This notebook applies Sentence Transformers to encode Arabic poetry into sentence embeddings, enabling efficient semantic similarity-based retrieval.

### 11. `11_gpt_search.ipynb`
This notebook demonstrates how GPT models can be used for semantic search by transforming queries and poetry into a more compatible search space.

### 12. `12_keyword_search.ipynb`
Provides a basic keyword-based search method as a comparison to the semantic search techniques, highlighting the differences and advantages of semantic understanding.

### 13. `13_last_word_predict.ipynb`
Focuses on a text generation task where the model predicts the next word in a given Arabic poetry line based on previous context.

### 14. `14_vector_db.ipynb`
This notebook explores the use of vector databases to store and query poetry embeddings, optimizing search performance for large-scale datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hawiya.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks: The project is organized in Jupyter notebooks, which can be run individually or in sequence. Ensure you have Jupyter installed, and launch the notebooks using:
   ```bash
   jupyter-notebook
   ```

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

You are free to share and adapt the material under the following terms:
* **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* **NonCommercial**: You may not use the material for commercial purposes.

For more details, see the LICENSE file or visit https://creativecommons.org/licenses/by-nc/4.0/.
