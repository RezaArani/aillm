// Copyright (c) 2025 John Doe
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
	"errors"

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
//   - title: The title associated with the content to be embedded.
//   - source: Source of selected data.

// Returns:
//   - []string: A slice of keys representing the stored embeddings in the vector database.
//   - int: The number of chunks the text was split into.
//   - error: An error if the embedding process fails.
func (llm *LLMContainer) embedText(prefix, contents, title, source string) ([]string, int, error) {
	var docList []string
	// Check if the embedding model is available

	if llm.Embedder == nil {
		return nil, 0, errors.New("missing embedding model")
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
	docs, splitErr := textEmbedding.SplitText()
	if splitErr != nil {
		return docList, 0, splitErr
	}

	// Add metadata to each chunk by prepending the title
	if source != "" {
		for idx, doc := range docs {
			doc.PageContent = "source: " + source + "\n" + doc.PageContent
			docs[idx] = doc
		}
	}
	if title != "" {
		for idx, doc := range docs {
			doc.PageContent = "Title: " + title + "\n" + doc.PageContent
			docs[idx] = doc
		}
	}

	// Get the embedding model from the initialized client
	embedder, err := llm.Embedder.NewEmbedder()
	if err != nil {
		return docList, 0, err
	}

	// Setup Redis vector store with index name and embedding model
	redisVector := redisvector.WithIndexName(prefix+"aillm_vector_idx", true)
	embedderVector := redisvector.WithEmbedder(embedder)

	// Retrieve Redis host URL for connection
	redisHostURL, redisConnectionErr := llm.getRedisHost()
	if redisConnectionErr != nil {
		return docList, 0, redisConnectionErr
	}

	// Create a new vector store using Redis and embedding model

	store, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), redisVector, embedderVector)
	if err != nil {
		return docList, 0, err
	}

	// Store the document chunks into the Redis vector store
	docLen := len(docs)
	if docLen > 0 {
		docList, err = store.AddDocuments(context.Background(), docs)
		if err != nil {
			return docList, 0, err
		}
	}

	return docList, docLen, nil
}
