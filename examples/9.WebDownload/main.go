package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	log.Println("Start:")

	embeddingllmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "all-minilm",
		},
	}

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "Meta-Llama-3_1-70B-Instruct",
			APIToken: os.Getenv("APITOKEN"),
		},
	}

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:  embeddingllmclient,
		LLMClient: llmclient,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()

	// Let's create an empty index:
	embedd(llm, "Hi")
	// asking  "What is SemMapas?" won't return any value
	askKLLM(llm,  "What is SemMapas?")

	// let's fetch and embed some data from https://semmapas.com
	tc := aillm.TranscribeConfig{
		Language: "en",
	}
	llm.EmbeddURL("", "https://semmapas.com", tc)
	// before
	askKLLM(llm, "What is SemMapas?")

	// Cleanup
	llm.RemoveEmbedding("")

}

func askKLLM(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + query + ":")
	queryResult, err := llm.AskLLM(query, llm.WithStreamingFunc(print))
	response := queryResult.Response
	resDocs := queryResult.RagDocs
	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
	log.Println("Reference Documents: ", len(resDocs.([]schema.Document)))

	for idx, doc := range resDocs.([]schema.Document) {
		srcDocs := fmt.Sprintf("\t%v. Score: %v,\tSource: %s+...", idx+1, doc.Score, doc.PageContent[:50])
		log.Println(srcDocs)
	}

}


func embedd(llm aillm.LLMContainer, Content string) {
	// Text Embedding
	contents:= aillm.LLMEmbeddingContent{
		Text: Content,
	}
	llm.EmbeddText("", contents)
}
func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
