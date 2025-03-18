package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Start:")

	// this example requires TIKA , for more information and installation instruction go to https://hub.docker.com/r/apache/tika

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

	// let's embed a PDF,Word, Excel, Powerpoint or... file

	llm.Transcriber.TikaURL = "http://localhost:9998"
	_, err := llm.EmbeddFile( "SampleIndex","Sample Title", llm.Transcriber.TempFolder+"/example.pdf", aillm.TranscribeConfig{
		Language:            "en",
		TikaLanguage:        "eng",
		ExtractInlineImages: true,
		MaxTimeout:          5 * time.Minute,
	})
	if err != nil {
		panic(err)
	}
	// Now it knows :-)
	askKLLM(llm,   "What is SemMapas?")
	// time to ask complex question
	askKLLM(llm,  "Sum of the costs?")

	// Cleanup
	llm.RemoveEmbedding("SampleIndex")

}

func askKLLM(llm aillm.LLMContainer,  query string) {
	log.Println("LLM Reply to " + query  + ":")
	queryResult, err := llm.AskLLM(query, llm.WithStreamingFunc(print))
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
 

func embedd(llm aillm.LLMContainer,  Content string) {
	// Text Embedding
	contents:= aillm.LLMEmbeddingContent{
		Text: Content,
	}
	llm.EmbeddText( "SemMapas", contents)
}
func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
