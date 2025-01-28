package main

import (
	"context"
	"fmt"
	"log"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	log.Println("Start:")

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
		DataRedis: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()
	embeddingId := "MyId"
	embeddingTitle := "rawText"
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingId)
	// looks for the data in "en" language contents
	askKLLM(llm, embeddingId, "en", "User1", "Tell me about SemMapas?")
	// looks for the data in "pt" language contents but replies in English
	askKLLM(llm, embeddingId, "en", "User2", "Tell me about JohnDoe?")

	// Information about SemMapas for user 1 based on memory 
	askKLLM(llm, embeddingId, "en", "User1", "Where and when?")
	// Information about JohnDoe for user 1 based on memory
	askKLLM(llm, embeddingId, "en", "User2", "Where and when?")

	//Let's delete User1's memory and ask the same question again.
	llm.MemoryManager.DeleteMemory("User1")
	askKLLM(llm, embeddingId, "en", "User1", "Where and when?")

	// Cleanup
	removeEmbedd(llm, embeddingId, embeddingTitle)

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
func embedd(llm aillm.LLMContainer, EmbeddingId string) {
	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: SemMapas,
	}
	
	llm.EmbeddText(EmbeddingId, "SemMapas", contents)
	contents = make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: JohnDoe,
	}
	llm.EmbeddText(EmbeddingId, "JohnDoe", contents)

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const SemMapas = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal.
`


const JohnDoe = `Welcome to JohnDoe, your trusted partner in delivering cutting-edge solutions for sustainable urban development and smart city innovations. At JohnDoe, we specialize in creating intelligent systems that empower municipalities and businesses to thrive in a rapidly evolving world.

Since our inception in 2018 in Berlin, Germany, JohnDoe has been at the forefront of revolutionizing urban spaces through tailored technology and data-driven insights. Designed with both public and private sectors in mind, our platform enables seamless integration of smart infrastructure, community engagement, and sustainable practices.

With JohnDoe, you can efficiently optimize city planning, monitor environmental impacts, and implement real-time solutions that enhance quality of life for citizens. Our intuitive tools and customizable features allow you to address the unique challenges of your city or business, ensuring lasting growth and innovation.
`
