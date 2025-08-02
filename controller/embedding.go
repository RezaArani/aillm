// Copyright (c) 2025 Reza Arani
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package aillm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/redisvector"
)

// LLMTextEmbedding is a struct designed to handle text processing and splitting operations.
//
// Fields:
//   - ChunkSize: The maximum size of each text chunk after splitting.
//   - ChunkOverlap: The number of overlapping characters between consecutive chunks to ensure context retention.
//   - Text: The original text content to be processed and split into chunks.
//   - EmbeddedDocuments: A slice of schema.Document representing the resulting chunks after processing.
type LLMTextEmbedding struct {
	ChunkSize         int
	ChunkOverlap      int
	Text              string
	EmbeddedDocuments []schema.Document
	lLMContainer      *LLMContainer // LLM container for embedding and vector search
}

// EmbeddingClient defines an interface to abstract embedding model clients.
//
// This interface provides methods to initialize and retrieve an embedding model,
// ensuring a standard contract for different embedding providers such as Ollama and OpenAI.
//
// Methods:
//   - NewEmbedder(): Initializes and returns an embedding model instance, or an error if the operation fails.
//   - initialized(): Checks if the embedding model has been initialized and is ready for use.
type EmbeddingClient interface {
	// NewEmbedder initializes and returns an embedding model instance.
	NewEmbedder() (embeddings.Embedder, error)
	// initialized checks whether the embedding model has been successfully initialized.
	initialized() bool
}

// InitEmbedding initializes the embedding model based on the type of embedding provider.
//
// This function checks the type of the configured embedding provider (Ollama or OpenAI)
// and initializes it with the appropriate configuration settings.
//
// Returns:
//   - error: Returns an error if initialization fails or if the provider is unsupported.
func (llm *LLMContainer) InitEmbedding() error {
	// Check the type of embedding provider and initialize accordingly
	switch llm.Embedder.(type) {
	// Initialize embedding for Ollama provider
	case *OllamaController:

		ollamaLLM, err := ollama.New(
			ollama.WithServerURL(llm.Embedder.(*OllamaController).Config.Apiurl),
			ollama.WithModel(llm.Embedder.(*OllamaController).Config.AiModel),
		)
		if err != nil {
			return err
		}
		// Assign the initialized Ollama instance to the controller
		llm.Embedder.(*OllamaController).LLMController = ollamaLLM
		// Initialize embedding for OpenAI provider

	case *OpenAIController:
		openaiLLM, err := openai.New(
			openai.WithToken(llm.Embedder.(*OpenAIController).Config.APIToken),
			openai.WithModel(llm.Embedder.(*OpenAIController).Config.AiModel),
		)
		if err != nil {
			return err
		}
		// Assign the initialized OpenAI instance to the controller
		llm.Embedder.(*OpenAIController).LLMController = openaiLLM

	default:
		// Handle unsupported embedding providers
		return errors.New("unsupported provider")
	}
	return nil
}

// embedText processes and embeds the given text content into a vector store.
//
// This function splits the input text into chunks, adds metadata, and stores it in a Redis-based vector database.
// It initializes the embedding model if not already initialized.
//
// Parameters:
//   - prefix: A string used as a prefix for storing the embedded content, typically indicating object context.
//   - contents: The text content to be embedded and stored in the vector store.
// 	 - title: The title associated with the content to be embedded inline with contents for better retrival.
//   - Index: The Index associated with the content to be embedded.
//   - source: Source of selected data.
//	 - GeneralEmbeddingDenied , prevents indexing in global search
//	 - rawKey, won't allow to process the index key automatically. (for some specific actions like memory search)

// Returns:
//   - []string: A slice of keys representing the stored embeddings in the vector database.
//   - int: The number of chunks the text was split into.
//   - error: An error if the embedding process fails.
func (llm *LLMContainer) embedText(prefix, language, index, title, contents string, sources string, metaData LLMEmbeddingContent, GeneralEmbeddingDenied, rawKey, useLLM bool) (docList []string, generalDocList []string, docLen int, inconsistentChunks map[int]string, err error) {
	// Check if the embedding model is available
	if llm.Embedder == nil {
		return nil, nil, docLen, inconsistentChunks, errors.New("missing embedding model")
	} else {
		// Initialize embedding model if it hasn't been initialized yet

		if !llm.Embedder.initialized() {
			llm.InitEmbedding()
		}
	}
	// Prepare the document text embedding configuration
	textEmbedding := LLMTextEmbedding{
		ChunkSize:    llm.EmbeddingConfig.ChunkSize,
		ChunkOverlap: llm.EmbeddingConfig.ChunkOverlap,
		Text:         contents,
	}

	// Split the text content into chunks
	docs, splitErr := []schema.Document{}, error(nil)

	if useLLM {
		var keywords []string
		docs, keywords, inconsistentChunks, splitErr = textEmbedding.SplitTextWithLLM()
		metaData.Keywords = keywords
	} else {
		docs, splitErr = textEmbedding.SplitText()
	}
	if splitErr != nil {

		return docList, generalDocList, docLen, inconsistentChunks, splitErr
	}

	// Add metadata to each chunk by prepending the source
	for idx, doc := range docs {
		// doc.PageContent = "source: " + source + "\n" + doc.PageContent
		doc.Metadata = make(map[string]any)
		metaData.Text = ""
		jsonMeta, _ := json.Marshal(metaData)
		doc.Metadata["rawkey"] = string(jsonMeta)
		doc.Metadata["sources"] = sources
		if title != "" {
			doc.PageContent = "Title: " + title + "\n" + doc.PageContent
		}
		if len(metaData.Keywords) > 0 {
			doc.PageContent += "\nKeywords: " + strings.Join(metaData.Keywords, ", ")
		}
		docs[idx] = doc
	}

	// Get the embedding model from the initialized client
	embedder, err := llm.Embedder.NewEmbedder()
	if err != nil {
		return docList, generalDocList, docLen, inconsistentChunks, splitErr
	}

	// Setup Redis vector store with index name and embedding model
	keyName := prefix
	if keyName != "" {
		keyName += ":"
	}
	keyName += index
	if !rawKey {
		keyName = "context:"
		if prefix != "" {
			keyName += prefix + ":"
		}
		keyName += index
		if language != "" {
			keyName += ":" + language
		}
		keyName += ":aillm_vector_idx"
	}
	redisVector := redisvector.WithIndexName(keyName, true)
	embedderVector := redisvector.WithEmbedder(embedder)

	// Retrieve Redis host URL for connection
	redisHostURL, redisConnectionErr := llm.getRedisHost()
	if redisConnectionErr != nil {
		return docList, generalDocList, docLen, inconsistentChunks, splitErr
	}

	// Create a new vector store using Redis and embedding model

	store, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), redisVector, embedderVector)
	if err != nil {
		return docList, generalDocList, docLen, inconsistentChunks, splitErr
	}

	// Store the document chunks into the Redis vector store
	docLen = len(docs)
	if docLen > 0 {
		docList, err = store.AddDocuments(context.Background(), docs)
		if err != nil {
			return docList, generalDocList, docLen, inconsistentChunks, splitErr
		}
		if !GeneralEmbeddingDenied && !rawKey {
			allKey := "all:"
			if prefix != "" {
				allKey += prefix + ":"
			}
			// allKey += index + ":"
			if language != "" {
				allKey += language + ":"
			}
			allKey += "aillm_vector_idx"
			generalRedisVector := redisvector.WithIndexName(allKey, true)
			generalStore, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), generalRedisVector, embedderVector)
			if err != nil {
				return docList, generalDocList, 0, inconsistentChunks, splitErr
			}

			generalDocList, err = generalStore.AddDocuments(context.Background(), docs)
			if err != nil {
				return docList, generalDocList, 0, inconsistentChunks, splitErr
			}
		}

	}

	return docList, generalDocList, docLen, inconsistentChunks, nil
}

// cleanEmbeddings cleans the embeddings from the Redis database.
//
// Parameters:
//   - Confirm: The confirmation string to clean the embeddings.
//   - prefix: The prefix of the embeddings to clean.
//   - index: The index of the embeddings to clean.
//
// Returns:
//   - error: An error if the cleaning fails.
func (llm *LLMContainer) CleanEmbeddings(Confirm, prefix string) error {
	if Confirm == "yes" {
		_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, "doc:all:"+prefix, true)
		if err != nil {
			return err
		}
		_, err = llm.deleteRedisWildCard(llm.RedisClient.redisClient, "doc:context:"+prefix, true)
		if err != nil {
			return err
		}
		_, err = llm.deleteRedisWildCard(llm.RedisClient.redisClient, "rawDocs:"+prefix, true)
		if err != nil {
			return err
		}

		res, err := llm.RedisClient.redisClient.Do(context.TODO(), "FT._LIST").Result()
		if err != nil {
			return err
		}

		// convert the result to a list of indexes
		indexes, ok := res.([]interface{})
		if !ok {
			return err
		}

		// delete indexes that match the wildcard
		err = llm.deleteIndexes(indexes, "context:"+prefix)
		if err != nil {
			return err
		}
		err = llm.deleteIndexes(indexes, "all:"+prefix)
		if err != nil {
			return err
		}
		err = llm.deleteIndexes(indexes, "rawDocsIdx:"+prefix)
		if err != nil {
			return err
		}

		//memory indexes should be implemented

	}

	return nil
}

func (llm *LLMContainer) deleteIndexes(indexes []interface{}, prefix string) error {
	for _, idx := range indexes {
		indexName := fmt.Sprintf("%v", idx)
		if strings.HasPrefix(indexName, prefix) {
			_, err := llm.RedisClient.redisClient.Do(context.TODO(), "FT.DROPINDEX", indexName, "DD").Result()
			if err != nil {
				return err
			}
		}
	}
	return nil
}
