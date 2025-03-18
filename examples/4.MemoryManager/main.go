package main

import (
	"context"
	"fmt"
	"log"

	aillm "github.com/RezaArani/aillm/controller"
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
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}

	llm.Init()
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	// looks for the data in "en" language contents
	askKLLM(llm, "User1", "Tell me about SemMapas?")
	// looks for the data in "pt" language contents but replies in English
	askKLLM(llm, "User2", "Tell me about JohnDoe?")

	// Information about SemMapas for user 1 based on memory 2023 in Portugal
	askKLLM(llm, "User1", "Where and when?")
	// Information about JohnDoe for user 1 based on memory Berlin, Germany, 2018
	askKLLM(llm, "User2", "Where and when?")

	//Let's delete User1's memory and ask the same question again. result should be something like :I'm unable to provide a specific location or time frame as I couldn't find any relevant information regarding your query.

	llm.MemoryManager.DeleteMemory("User1")
	askKLLM(llm, "User1", "Where and when?")

	// Cleanup
	llm.RemoveEmbedding("SemMapas")

}

func askKLLM(llm aillm.LLMContainer, user, query string) {
	log.Println("LLM Reply to " + query + " from " + user + ":")
	queryResult, err := llm.AskLLM(query, llm.WithSessionID(user), llm.WithStreamingFunc(print))
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

func embedd(llm aillm.LLMContainer) {
	// Text Embedding
	
	contents:= aillm.LLMEmbeddingContent{
		Text: SemMapas,
		Language: "en",
	}
	llm.EmbeddText("SemMapas", contents)

	contents = aillm.LLMEmbeddingContent{
		Text: JohnDoe,
		Language: "en",
	}
	llm.EmbeddText("SemMapas", contents)
 
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
