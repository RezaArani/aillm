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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// DescribeImage sends an image along with a text query to an AI vision model for description.
//
// This function constructs a JSON payload containing the provided image (Base64-encoded) and query,
// then sends an HTTP POST request to the AI model endpoint. The API response is processed to extract
// the textual description of the image.
//
// Parameters:
//   - encodedImage: A string containing the Base64-encoded image data.
//   - query: A text prompt to know how to deal with the image (e.g., "Describe this image.").
//   - options: Variadic LLMCallOption parameters for additional configuration (if applicable).
//
// Returns:
//   - string: The textual description generated by the AI model.
//   - error: An error if the request fails or the response is invalid.
//
// Example Usage:
//
//	description, err := llm.DescribeImage(encodedImage, "Describe this image.")
//	if err != nil {
//	    log.Fatalf("Error: %v", err)
//	}
//	fmt.Println("Image Description:", description)
//
// Notes:
//   - The function extracts the AI model and API details from `llm.VisionClient.GetConfig()`.
//   - It expects a valid API token to be set in `llm.VisionClient.GetConfig().APIToken`.
//   - The function handles HTTP request creation, response validation, and JSON parsing.
//   - If the API response structure changes, parsing logic might require adjustments.
func (llm *LLMContainer) DescribeImage(encodedImage, query string, options ...LLMCallOption) (string, error) {

	// setup payload JSON
	payload := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"name": "User",
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type": "text",
						"text": query,
					},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": encodedImage,
						},
					},
				},
			},
		},
		"model":       llm.VisionClient.GetConfig().AiModel,
		"temperature": llm.Temperature,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("error serialization of JSON: %v", err)
	}

	// HTTP request
	req, err := http.NewRequest("POST", llm.VisionClient.GetConfig().Apiurl+"chat/completions", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return "", fmt.Errorf("error creating http request, HTTP: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", llm.VisionClient.GetConfig().APIToken))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error calling ai provider: %v", err)
	}
	defer resp.Body.Close()

	// API status check
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("خطا در پاسخ API: %d", resp.StatusCode)
	}

	// reading JSON
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body from API: %v", err)
	}

	// process JSON response
	var responseData map[string]interface{}
	if err := json.Unmarshal(body, &responseData); err != nil {
		return "", fmt.Errorf("error processing JSON response: %v", err)
	}

	// extracting API response
	if choices, found := responseData["choices"].([]interface{}); found {
		for _, choice := range choices {
			choiceMap := choice.(map[string]interface{})
			if message, ok := choiceMap["message"].(map[string]interface{}); ok {
				if content, exists := message["content"].(string); exists {
					return content, nil
				}
			}
		}
	}

	return "", fmt.Errorf("error extracting response from API")
}

// DescribeImageFromFile reads an image file, encodes it to Base64, and sends it along with a text query
// to an AI vision model for description.
//
// This function reads an image file from the specified path, converts it to a Base64-encoded string,
// and calls the `DescribeImage` function to obtain a textual description of the image.
//
// Parameters:
//   - imagePath: The file path of the image to be processed.
//   - query: A text query to describe the image (e.g., "Describe this image.").
//   - options: Variadic LLMCallOption parameters for additional configuration (if applicable).
//
// Returns:
//   - string: The textual description generated by the AI model.
//   - error: An error if the file cannot be read, encoding fails, or the AI request encounters an issue.
//
// Example Usage:
//
//	description, err := llm.DescribeImageFromFile("sample.jpg", "Describe this image.")
//	if err != nil {
//	    log.Fatalf("Error: %v", err)
//	}
//	fmt.Println("Image Description:", description)
//
// Notes:
//   - The function reads the image as a binary file using `os.ReadFile()`.
//   - It converts the image to a Base64-encoded string before calling `DescribeImage()`.
//   - Requires a valid API token and properly configured `llm.VisionClient` settings.
//   - If the image file is too large, encoding may consume significant memory.
func (llm *LLMContainer) DescribeImageFromFile(imagePath, query string, options ...LLMCallOption) (string, error) {
	// reading image file
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("error reading file: %v", err)
	}
	// converting image to base64 string
	encodedImage := fmt.Sprintf("data:;base64,%s", base64.StdEncoding.EncodeToString(imageData))
	return llm.DescribeImage(encodedImage, query, options...)
}
