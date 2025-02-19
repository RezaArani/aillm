package main

import (
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
)

// main initializes the AI model and sends an image for description.
//
// This function sets up an AI model client using `aillm.OpenAIController`, configures the API parameters,
// and then calls the `Describe` function to process an image. The expected output is a detailed textual
// description of the image.
//
// Notes:
//   - The AI model endpoint and API token are read from the environment variable `APITOKEN`.
//   - The expected response is a structured natural language description of the image content.
//   - The function logs the AI-generated description to the console.
func main() {
	log.Println("Testing aillm framework:")

	// Initialize AI Vision Client with configuration

	visionclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://llava-next-mistral-7b.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "llava-next-mistral-7b",
			APIToken: os.Getenv("APITOKEN"), // API token should be set in environment variables
		},
	}
	// Create LLMContainer instance with the vision client and a confidence score threshold

	llm := aillm.LLMContainer{
		VisionClient:   visionclient,
		ScoreThreshold: 0.6,
	}
	// Call Describe function to analyze an image

	//Response should be The image shows a plate of sliced beef, likely a cut such as ribeye, served with a side of greens, possibly arugula salad. The beef is cooked to a medium-rare doneness, with a rich, dark color indicating it is well-seared. The plate is placed on a wooden cutting board, and there is a small bowl with what appears to be a sauce or condiment, possibly a horseradish cream or a herb butter. To the right of the plate, there is a wooden spoon with a few whole peppercorns on it, suggesting they might be used to add flavor to the dish. In the background, there is a lemon wedge, which could be used to add a citrus note to the meal. The overall setting suggests a restaurant or a well-presented home-cooked meal. The lighting is warm and soft, enhancing the colors and textures of the food.
	Describe(llm, "Describe it.")

}

// Describe sends an image and query to the AI model for processing.
//
// This function calls `DescribeImage` from `LLMContainer`, passing an image file (`steak.jpg`) 
// and a text prompt to generate a description. The response is logged to the console.
//
// Parameters:
//   - llm: The AI-powered language model container (`LLMContainer`) used for processing requests.
//   - query: A text query instructing the AI on what to describe in the image.
//
// Returns:
//   - Logs the AI-generated description of the image or exits with an error if the request fails.
//
// Example Usage:
//   Describe(llm, "Describe this dish.")
//
// Notes:
//   - The function expects `steak.jpg` to be present in the working directory.
//   - If the API request fails, the function logs the error and exits.
func Describe(llm aillm.LLMContainer, query string) {
	log.Println("Model Reply to " + query + ":")

	// Call AI model to describe the image
	imageDescription, err := llm.DescribeImageFromFile("steak.jpg", query)
	if err != nil {
		log.Fatal(err)
	}
	// Log the response from AI model
	log.Println(imageDescription)

}
