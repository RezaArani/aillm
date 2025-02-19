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
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

// OllamaController struct to manage the Ollama embedding and language model service.
//
// This struct implements the EmbeddingClient interface and acts as a wrapper around
// the Ollama LLM (Large Language Model), handling initialization and interactions.
//
// Fields:
//   - Config: Configuration details such as API URL and model name.
//   - LLMController: Instance of the Ollama LLM client for handling AI operations.
type OllamaController struct {
	Config        LLMConfig   // Configuration for the Ollama LLM service
	LLMController *ollama.LLM // Instance of the Ollama LLM client
}

// NewEmbedder initializes and returns an Ollama embedding model instance.
//
// This function implements the EmbeddingClient interface to create and return an embedding model
// using the current LLMController instance.
//
// Returns:
//   - embeddings.Embedder: The initialized embedding model instance.
//   - error: An error if the initialization fails.
func (oc *OllamaController) NewEmbedder() (embeddings.Embedder, error) {
	return embeddings.NewEmbedder(oc.LLMController)
}

// NewLLMClient initializes and returns a new instance of the Ollama LLM client.
//
// This function sets up the Ollama model based on the provided API URL and model name
// in the configuration.
//
// Returns:
//   - llms.Model: The initialized LLM model instance.
//   - error: An error if the initialization fails.
func (oc *OllamaController) NewLLMClient() (llms.Model, error) {
	var err error
	oc.LLMController, err = ollama.New(ollama.WithServerURL(oc.Config.Apiurl), ollama.WithModel(oc.Config.AiModel))
	return oc.LLMController, err
}

// initialized checks if the Ollama LLM client has been successfully initialized.
//
// This function returns a boolean value indicating whether the LLMController has been
// successfully instantiated.
//
// Returns:
//   - bool: True if the LLMController is initialized, otherwise false.
func (oc *OllamaController) initialized() bool {
	return oc.LLMController != nil
}

func (oc *OllamaController) GetConfig() LLMConfig {
	return oc.Config
}
