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
	"github.com/tmc/langchaingo/llms/openai"
)

// OpenAIController struct to manage OpenAI embedding and language model services.
//
// This struct implements the EmbeddingClient interface and acts as a wrapper around
// the OpenAI LLM (Large Language Model), handling initialization and interactions.
//
// Fields:
//   - Config: Configuration details such as API URL, model name, and API token.
//   - LLMController: Instance of the OpenAI LLM client for handling AI operations.
type OpenAIController struct {
	Config        LLMConfig
	LLMController *openai.LLM
}

// NewEmbedder initializes and returns an OpenAI embedding model instance.
//
// This function implements the EmbeddingClient interface to create and return an embedding model
// using the current LLMController instance.
//
// Returns:
//   - embeddings.Embedder: The initialized embedding model instance.
//   - error: An error if the initialization fails.
func (oc *OpenAIController) NewEmbedder() (embeddings.Embedder, error) {
	return embeddings.NewEmbedder(oc.LLMController)
}

// NewLLMClient initializes and returns a new instance of the OpenAI LLM client.
//
// This function sets up the OpenAI model based on the provided API token, API base URL,
// and the selected AI model from the configuration.
//
// Returns:
//   - llms.Model: The initialized LLM model instance.
//   - error: An error if the initialization fails.
func (oc *OpenAIController) NewLLMClient() (llms.Model, error) {
	var err error
	oc.LLMController, err = openai.New(openai.WithToken(oc.Config.APIToken), openai.WithBaseURL(oc.Config.Apiurl), openai.WithModel(oc.Config.AiModel), openai.WithEmbeddingModel(oc.Config.AiModel))
	//  openai.New(openai.WithToken(oc.Config.APIToken), openai.WithBaseURL(oc.Config.Apiurl), openai.WithModel(oc.Config.AiModel))
	return oc.LLMController, err
}

// initialized checks if the OpenAI LLM client has been successfully initialized.
//
// This function returns a boolean value indicating whether the LLMController has been
// successfully instantiated.
//
// Returns:
//   - bool: True if the LLMController is initialized, otherwise false.
func (oc *OpenAIController) initialized() bool {
	return oc.LLMController != nil
}

func (oc *OpenAIController) GetConfig() LLMConfig {
	return oc.Config
}
