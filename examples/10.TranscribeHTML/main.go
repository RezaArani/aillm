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
		DataRedis: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()
	embeddingId := "MyId"
	embeddingTitle := "rawText"

	// Let's create an empty index:
	embedd(llm, embeddingId,embeddingTitle,"Hi")
	// asking  "What is SemMapas?" won't return any value
	askKLLM(llm, embeddingId, "en", "User1", "What is SemMapas?")

	// let's embed a html file
	
	EmbeddingId := "MyId"
 
	llm.EmbeddFile(EmbeddingId, "SampleFile", llm.Transcriber.TempFolder+"\\semmapas.html", aillm.TranscribeConfig{
		Language:     "en",
	})
	
	// Now it knows :-)
	askKLLM(llm, embeddingId, "en", "User1", "What is SemMapas?")
	// looks for the data in "pt" language contents but replies in English

	// Cleanup
	removeEmbedd(llm, embeddingId, embeddingTitle)
	removeEmbedd(llm, embeddingId, "")

}

func askKLLM(llm aillm.LLMContainer, EmbeddingId, Language, user, query string) {
	log.Println("LLM Reply to " + query + " from " + user + ":")
	response, resDocs, err := llm.AskLLM(EmbeddingId, Language, user, query, print)
	if err!=nil{
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

func removeEmbedd(llm aillm.LLMContainer, embeddingId, title string) {
	llm.RemoveEmbeddingDataFromRedis(embeddingId, title)
}

func embedd(llm aillm.LLMContainer, EmbeddingId , EmbeddingTitle,Content string) {
	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: Content,
	}
	llm.EmbeddText(EmbeddingId, "SemMapas", contents)
}
func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
