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
	"errors"

	"github.com/redis/go-redis/v9"
	"github.com/tmc/langchaingo/llms"
)

// LLMConfig struct holds configuration details for the embedding and AI model service.
//
// This struct is used to store the necessary configuration parameters needed to interact
// with LLM (Large Language Model) services, including API endpoint URLs, model names,
// and authentication credentials.
//
// Fields:
//   - Apiurl: The API endpoint URL of the LLM service for sending requests.
//   - AiModel: The specific AI model to be used for embedding or inference operations.
//   - APIToken: Authentication token required to access the API (e.g., for OpenAI services).
type LLMConfig struct {
	Apiurl   string // API endpoint for the LLM service
	AiModel  string // Name of the AI model to be used
	APIToken string // API key required for authorization (e.g., for OpenAI or OVHCloud)
}

// LLMClient defines an interface for creating a new LLM (Large Language Model) client instance.
//
// Methods:
//   - NewLLMClient(): Creates and returns an instance of an LLM model, or returns an error if the initialization fails.
type LLMClient interface {
	// NewLLMClient initializes and returns an LLM model instance.
	NewLLMClient() (llms.Model, error)
}

// EmbeddingConfig holds the configuration settings for text chunking during embedding operations.
//
// Fields:
//   - ChunkSize: The size of each chunk to be created when splitting text for embedding purposes.
//   - ChunkOverlap: The number of overlapping characters between consecutive chunks to maintain context.
type EmbeddingConfig struct {
	ChunkSize    int // Size of each text chunk for embedding
	ChunkOverlap int // Number of overlapping characters between chunks
}

// RedisClient manages the connection details for a Redis database instance used for storing embeddings.
//
// Fields:
//   - Host: The address of the Redis server (e.g., "localhost:6379").
//   - Password: The password for connecting to the Redis server (if authentication is required).
//   - redisClient: The Redis client instance used for executing operations.
type RedisClient struct {
	Host        string        // Redis server address
	Password    string        // Redis authentication password (if applicable)
	redisClient *redis.Client // Redis client instance for operations
}

const (
	SimilaritySearch  = 1 //
	KNearestNeighbors = 2 //
)

// LLMContainer serves as the main struct that manages LLM operations, embedding configurations, and data storage.
//
// It acts as a container for managing various components required for interacting with
// an AI model, embedding data, and handling queries and responses.
//
// Fields:
//   - Embedder: The embedding client responsible for processing and storing text embeddings.
//   - EmbeddingConfig: Configuration settings for text chunking operations.
//   - LLMClient: The LLM client that provides access to the AI model for generating responses.
//   - MemoryManager: A memory management component that stores session-related data.
//   - LLMModelLanguageDetectionCapability: A boolean indicating if the model supports automatic language detection.
//   - AnswerLanguage: The preferred language for responses from the model.
//   - DataRedis: Redis client for caching embeddings and retrieval operations.
//   - Temperature: Controls the randomness of the AI's responses (lower values = more deterministic).
//   - TopP: Probability threshold for response generation (higher values = more diverse responses).
//   - ScoreThreshold: The similarity threshold for retrieval-augmented generation (RAG).
//   - RagRowCount: The number of RAG rows to retrieve and analyze for context.
//   - AllowHallucinate: Determines if the model can generate responses without relevant data (true/false).
//   - FallbackLanguage: The default language to use if the primary language is unavailable.
//   - NoRagErrorMessage: The error message to display if no relevant data is found during retrieval.
//   - NotRelatedAnswer: A predefined response when the model cannot find relevant information.
//   - Character: A personality trait or characteristic assigned to the AI assistant (e.g., formal, friendly).
//   - Transcriber: Component responsible for converting speech or text inputs into usable data.
type LLMContainer struct {
	Embedder                            EmbeddingClient // Embedding client to handle text processing
	EmbeddingConfig                     EmbeddingConfig // Configuration for text chunking
	LLMClient                           LLMClient       // AI model client for generating responses
	MemoryManager                       *MemoryManager  // Session-based memory management
	LLMModelLanguageDetectionCapability bool            // Language detection capability flag
	AnswerLanguage                      string          // Default answer language - will be ignored if  LLMModelLanguageDetectionCapability = true
	DataRedis                           RedisClient     // Redis client for caching and retrieval
	SearchAlgorithm                     int             // Semantic search algorithm Cosine Similarity or The k-nearest neighbors
	Temperature                         float64         // Controls randomness of model output
	TopP                                float64         // Probability threshold for response diversity
	ScoreThreshold                      float32         // Threshold for RAG-based responses
	RagRowCount                         int             // Number of RAG rows to retrieve for context
	AllowHallucinate                    bool            // Enables/disables AI-generated responses when data is
	FallbackLanguage                    string          // Default language fallback
	NoRagErrorMessage                   string          // Message shown when RAG results are empty
	NotRelatedAnswer                    string          // Predefined response for unrelated queries
	Character                           string          // AI assistant's character/personality settings
	Transcriber                         Transcriber     // Responsible for processing and transcribing content
}

// getRedisHost constructs the Redis connection URL based on the stored Redis host and password.
//
// This function checks if the Redis host is set, and if so, it constructs a connection string
// with or without authentication credentials.
//
// Returns:
//   - string: A formatted Redis connection URL (e.g., "redis://localhost:6379").
//   - error: An error if the Redis host is not set.
func (llm *LLMContainer) getRedisHost() (string, error) {
	var err error
	host := ""

	// Check if the Redis host is set in the configuration
	
	if llm.DataRedis.Host == "" {
		err = errors.New("RedisHost is not set")
	} else {
		// Construct Redis connection string without authentication

		host = "redis://" + llm.DataRedis.Host

		// If password is provided, include it in the connection string

		if llm.DataRedis.Password != "" {
			host = "redis://:" + llm.DataRedis.Password + "@" + llm.DataRedis.Host
		}
	}

	return host, err
}
