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
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	// Distance to the result. greater means more R data but not accurate
	// llm.ScoreThreshold = 0.8
	askKLLM(llm,  "SemMapas city?")

	// A higher score results in the inclusion of more information, which increases the size of the RAG dataset and the number of tokens used. This can lead to several problems:
	// 1. Inclusion of excessive and irrelevant information can slow down response times, increase processing overhead on the LLM, and dilute the relevance of the output.
	// 2. Higher token usage for both the prompt and completion not only raises operational costs but also risks exceeding token limits, which may truncate outputs or lead to incomplete responses.
	// 3. Overloading the model with unnecessary data can reduce the overall accuracy and coherence of the generated content, affecting user experience and system efficiency.
	// It is essential to balance the score to optimize the trade-off between information richness and processing efficiency.

	llm.ScoreThreshold = 0.9
	// there are multiple nonrelated documents will be retrieved. It means our prompt will be bigger and less efficient
	askKLLM(llm,  "SemMapas city?")
	// Now let's make it more accurate
	llm.ScoreThreshold = 0.5
	// It just returns one row for the context
	askKLLM(llm, "SemMapas city?")

	// ask about another subject
	llm.ScoreThreshold = 0.7

	askKLLM(llm, "tell me about historical paris and farm issues?")

	// Tip for setting a good threshold for cosine similarity search:
	// Choose a threshold that ensures relevance while minimizing noise. Start with a threshold around 0.7–0.8, as this typically captures meaningful matches without overloading the response with less relevant data. Adjust based on your specific use case and test the quality of the retrieved results. Lower thresholds may retrieve irrelevant results, while higher thresholds might omit useful information.
	//
	// Tip on embedding models:
	// The quality of the embedding model used in cosine similarity search significantly impacts the results. Choosing a high-quality model that generates embeddings suited to your domain (e.g., general-purpose models like Sentence Transformers for diverse text or domain-specific embeddings for specialized tasks) can improve the accuracy of similarity matching. A good embedding model will better capture semantic meaning, allowing you to set more reliable thresholds and retrieve more relevant and concise results.

	// Cleanup
	llm.RemoveEmbedding("")
}

func askKLLM(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + ":")
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
 
func embedd(llm aillm.LLMContainer) {
	// Text Embedding
	contents := aillm.LLMEmbeddingContent{
		Text: SemMapas,
	}

	llm.EmbeddText( "", contents)

	contents = aillm.LLMEmbeddingContent{
		Text: JohnDoe,
	}
	llm.EmbeddText(  "", contents)

	
	contents = aillm.LLMEmbeddingContent{
		Text: ParisHistory,
	}
	llm.EmbeddText("", contents)

	contents = aillm.LLMEmbeddingContent{
		Text: CropPestInfection,
	}
	llm.EmbeddText( "", contents)

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const SemMapas = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal and city of Lourinhã.
`

const JohnDoe = `Welcome to JohnDoe, your trusted partner in delivering cutting-edge solutions for sustainable urban development and smart city innovations. At JohnDoe, we specialize in creating intelligent systems that empower municipalities and businesses to thrive in a rapidly evolving world.

Since our inception in 2018 in Berlin, Germany, JohnDoe has been at the forefront of revolutionizing urban spaces through tailored technology and data-driven insights. Designed with both public and private sectors in mind, our platform enables seamless integration of smart infrastructure, community engagement, and sustainable practices.

With JohnDoe, you can efficiently optimize city planning, monitor environmental impacts, and implement real-time solutions that enhance quality of life for citizens. Our intuitive tools and customizable features allow you to address the unique challenges of your city or business, ensuring lasting growth and innovation.
`

const ParisHistory = `Paris, often referred to as the "City of Light," boasts a rich history that spans over two millennia. Founded around 250 BC by a Celtic tribe called the Parisii, the city began as a small settlement on the Île de la Cité, a strategic location on the Seine River. Conquered by the Romans in 52 BC, it was renamed Lutetia and gradually evolved into a significant urban center. By the Middle Ages, Paris had become a thriving hub of commerce, education, and culture, and it was the birthplace of the University of Paris in the 12th century. Gothic architectural masterpieces like Notre-Dame Cathedral and Sainte-Chapelle reflect the city's medieval grandeur.
Paris continued to play a central role in European history through the Renaissance, the French Revolution, and the Napoleonic era, each leaving an indelible mark on the city. The Revolution transformed it into a symbol of liberty and modern democracy, while the 19th century saw the city’s transformation under Baron Haussmann, who modernized its layout with wide boulevards and grand parks. As a global epicenter of art, fashion, and intellectual thought, Paris has hosted world-changing movements and events, including the Impressionist movement and the 1889 World's Fair, which introduced the iconic Eiffel Tower. Today, Paris remains a vibrant blend of its storied past and its forward-looking cultural and economic influence. We love french agriculture industry.
`

const CropPestInfection = `Crop pest infections pose a significant threat to global agriculture, affecting both food security and economic stability. Pests such as insects, fungi, nematodes, and weeds can devastate crops by feeding on plants, spreading diseases, or competing for nutrients. Common pests like locusts, aphids, and armyworms can quickly infest fields, leading to severe yield losses, particularly in regions heavily reliant on agriculture. For example, locust swarms have been known to destroy thousands of hectares of crops within days, causing famine and financial hardship. Factors such as climate change, monoculture farming, and the misuse of pesticides have exacerbated pest outbreaks, creating challenges for farmers worldwide.
Addressing crop pest infections requires an integrated pest management (IPM) approach, combining cultural, biological, and chemical methods to minimize damage. Practices such as crop rotation, resistant crop varieties, and natural predators like ladybugs and parasitic wasps help control pest populations sustainably. Innovations in technology, including drone surveillance, precision agriculture, and biopesticides, have further enhanced farmers' ability to detect and combat pests early. However, global collaboration, research, and education are essential to develop scalable solutions, particularly for smallholder farmers in vulnerable regions. By investing in these strategies, it is possible to mitigate the impact of pest infections and ensure a stable food supply for growing populations. There are a lot of farms exists nearby Paris.`
