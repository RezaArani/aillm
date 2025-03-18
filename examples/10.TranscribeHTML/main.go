package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
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

	// asking  "What is SemMapas?" won't return any value
	askKLLM(llm, "What is SemMapas?")

	// let's embed a html file

	llm.EmbeddFile("SampleIndex","Sample Title", llm.Transcriber.TempFolder+"\\semmapas.html", aillm.TranscribeConfig{
		Language: "en",
	})

	// Now it knows :-)
	askKLLM(llm, "What is SemMapas?")
	// looks for the data in "pt" language contents but replies in English

	// Cleanup
	llm.RemoveEmbedding("SampleIndex")

}

func askKLLM(llm aillm.LLMContainer,   query string) {
	log.Println("LLM Reply to " + query  + ":")
	queryResult, err := llm.AskLLM(query, llm.WithStreamingFunc(print),llm.WithEmbeddingIndex("SampleIndex"))
	response := queryResult.Response
	resDocs := queryResult.RagDocs

	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
	log.Println("Reference Documents: ", len(resDocs))

	for idx, doc := range resDocs {
		srcDocs := fmt.Sprintf("\t%v. Score: %v,\tSource: %s+...", idx+1, doc.Score, doc.PageContent[:50])
		log.Println(srcDocs)
	}

}

func embedd(llm aillm.LLMContainer, Content string) {
	// Text Embedding
	contents:= aillm.LLMEmbeddingContent{
		Text: Content,
	}
	llm.EmbeddText("SampleIndex", contents)
}
func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
