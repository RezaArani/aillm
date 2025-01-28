package main

import (
	"context"
	"fmt"
	"log"
	"time"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	log.Println("Start:")
	log.Println(time.Now())
	// locally hosted ollama
	llmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "llama3.1",
		},
	}
 
	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:  llmclient,
		LLMClient: llmclient,
		ScoreThreshold: 0.8,
		DataRedis: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()
	embeddingId := "MyId"
	embeddingTitle := "rawText"
 	
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingId, embeddingTitle)
	// looks for the data in "en" language contents
	askKLLM(llm, embeddingId, "en", "User1", "What is SemMapas?")
	askKLLM(llm, embeddingId, "en", "User1", "Where did it launched?")
	// Now removing embedded data and asking the same question, result should be I'm unable to provide a specific location regarding the launch of SemMapas as I don't have sufficient information on this topic.
	removeEmbedd(llm, embeddingId, embeddingTitle)
	// looks for the data in "en" language contents
	askKLLM(llm, embeddingId, "en", "User1", "What is SemMapas?")
	askKLLM(llm, embeddingId, "en", "User1", "Where did it launched?")
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
func embedd(llm aillm.LLMContainer, EmbeddingId, Title string) {
	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: enRawText,
	}
 
	llm.EmbeddText(EmbeddingId, Title, contents)

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const enRawText = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal.
`
 