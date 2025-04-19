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
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"github.com/tmc/langchaingo/schema"
)

// LLMEmbeddingContent represents a single piece of text content that is embedded and stored in the system.
//
// This struct holds the necessary information for managing and identifying embedded text,
// including the raw text, a title, its source, and associated keys.
//
// Fields:
//   - Text: The raw text content that is embedded.
//   - Title: A descriptive title for the embedded content.
//   - Index:
//   - Source: The origin of the content, such as a file name, URL, or other identifier.
//   - Keys: A slice of strings representing the Redis keys associated with this content.
type LLMEmbeddingContent struct {
	Text        string   `json:"Text" redis:"Text"`
	Title       string   `json:"Title" redis:"Title"`
	Language    string   `json:"Language" redis:"Language"`
	Id          string   `json:"Id" redis:"Id"`
	Keys        []string `json:"Keys" redis:"Keys"`
	GeneralKeys []string `json:"GeneralKeys" redis:"GeneralKeys"`
	Keywords    []string `json:"Keywords" redis:"Keywords"`
	Sources     string   `json:"Sources" redis:"Sources"`
}

// LLMEmbeddingObject represents a collection of embedded text contents grouped under a specific object ID.
//
// This struct serves as a container for multiple pieces of embedded text content, organized by language or context.
// It provides a way to store and manage embeddings for a specific use case or document.
//
// Fields:
//   - EmbeddingPrefix: A unique prefix or identifier for the embedding object (e.g., "ObjectId").
//   - Index: An Index for the embedding object, providing a future access.
//   - Contents: A map of language-specific content, where the key is the language code (e.g., "en", "pt")
//     and the value is an LLMEmbeddingContent struct containing the associated content details.
type LLMEmbeddingObject struct {
	EmbeddingPrefix string                         `json:"EmbeddingPrefix" redis:"EmbeddingPrefix"`
	Index           string                         `json:"Index" redis:"Index"`
	Contents        map[string]LLMEmbeddingContent `json:"Contents" redis:"Contents"`
}

 
// getRawDocRedisId generates a unique Redis key for storing raw document data.
// It combines the object ID and a sanitized version of the Index to create a consistent key format.
//
// Returns:
//   - A string representing the Redis key in the format "rawDocs:ObjectId:Index".
func (llmeo LLMEmbeddingObject) getRawDocRedisId() string {
	// Construct Redis key using object ID and sanitized Index
	key := "rawDocs:"
	if llmeo.EmbeddingPrefix != "" {
		key += llmeo.EmbeddingPrefix + ":"
	}
	key += llmeo.sanitizeRedisKey(llmeo.Index)
	return key
}

// EmbeddFile processes and embeds the content of a given file into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Title: The Title of the document being embedded. Also it will be used for raw data for a better Context
//   - Index: The Index of the document being embedded. Also it will be used for the raw file redis key
//   - fileName: The path to the file to be embedded.
//   - tc: Configuration for transcription, such as language settings.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during processing.
func (llm LLMContainer) EmbeddFile(Index, Title, fileName string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	// EmbeddingContents := make(map[string]LLMEmbeddingContent)
	// Transcribe the file to extract text content
	fileContents, _, transcribeErr := llm.Transcriber.transcribeFile(fileName, "", tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with language as key
	EmbeddingContents := LLMEmbeddingContent{
		Text:    fileContents,
		Title:   Title,
		Sources: fileName,
	}

	// Embed the transcribed text into the LLM system
	embeddedTextObjects, embedErr := llm.EmbeddText(Index, EmbeddingContents, options...)
	if embedErr != nil {
		return result, embedErr
	}
	return embeddedTextObjects, embedErr
}

// EmbeddURL processes and embeds content from a given URL into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Index: The Index associated with the content being embedded.
//   - url: The web URL from which content will be transcribed and embedded.
//   - tc: Configuration for transcription, including language and extraction options.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during the transcription or embedding process.
func (llm LLMContainer) EmbeddURL(Index, url string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	// Transcribe the content from the provided URL
	fileContents, _, transcribeErr := llm.Transcriber.TranscribeURL(url, tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with the specified language as key
	EmbeddingContents := LLMEmbeddingContent{
		Text:    fileContents,
		Sources: url,
	}

	// Embed the transcribed text into the LLM system
	// o := LLMCallOptions{}
	// for _, opt := range options {opt(&o)}
	embeddedTextObjects, embedErr := llm.EmbeddText(Index, EmbeddingContents, options...)
	if embedErr != nil {
		return result, embedErr
	}
	// redisErr := llm.saveEmbeddingDataToRedis(embeddedTextObjects)
	return embeddedTextObjects, embedErr

}

// EmbeddText embeds provided text content into the LLM system and stores the data in Redis.
//
// Parameters:
//   - ObjectId: Unique identifier for the embedding object.
//   - Index: Index associated with the content being embedded.
//   - Contents: A map containing language-specific content to be embedded.
//
// Returns:
//   - LLMEmbeddingObject: The resulting embedding object after processing and storage.
//   - error: An error if any issues occur during embedding or Redis operations.
func (llm *LLMContainer) EmbeddText(Index string, Contents LLMEmbeddingContent, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}

	result := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}
	ctx := context.TODO()
	_, err := llm.RedisClient.redisClient.Ping(ctx).Result()
	if err != nil {
		return result, err
	}
	
	
	// Load existing data from Redis if available
	err = result.load(llm.RedisClient.redisClient, result.getRawDocRedisId())
	if err != nil && err.Error() != "key not found" {
		return result, err
	}

	// Process embedding for each language in contents
	if result.Contents == nil {
		result.Contents = make(map[string]LLMEmbeddingContent)
	}
	if Contents.Id == "" {
		Contents.Id = uuid.New().String()
	}
	//
	if o.CotextCleanup {
		Contents.Text = llm.Transcriber.cleanupText(Contents.Text, true)
	}
	if Contents.Language == "" {
		Contents.Language = o.Language
	}
	tempKeys, generalKeys, _, _, err := llm.embedText(o.getEmbeddingPrefix(), Contents.Language, Index, Contents.Title, llm.Transcriber.cleanupText(Contents.Text, o.CotextCleanup), Contents.Sources, Contents, o.LimitGeneralEmbedding, false, o.UseLLMToSplitText)
	if err != nil {
		return result, err
	}
	curContents := result.Contents[Contents.Id]
	// Cleanup previous keys
	for _, key := range curContents.Keys {
		llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
	}
	for _, key := range curContents.GeneralKeys {
		llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
	}

	// updating with new keys
	// tmpGeneralKeys := append(curContents.GeneralKeys, generalKeys...)
	// tmpKeys := append(curContents.GeneralKeys, tempKeys...)

	curContents = Contents
	curContents.GeneralKeys = generalKeys
	curContents.Keys = tempKeys

	result.Contents[Contents.Id] = curContents

	// Save the embedding data to Redis
	redisErr := llm.saveEmbeddingDataToRedis(result)
	return result, redisErr
}

// Load retrieves an embedding object from Redis storage based on a key.
//
// Parameters:
//   - client: Redis client instance for database operations.
//   - KeyID: The key used to retrieve the embedding object from Redis.
//
// Returns:
//   - error: An error if the key is not found or data cannot be unmarshalled.
func (llmEO *LLMEmbeddingObject) load(client *redis.Client, KeyID string) error {

	ctx := context.Background()

	// Retrieve data from Redis using the provided key
	data, err := client.Do(ctx, "JSON.GET", KeyID).Result()

	if err == redis.Nil {
		return fmt.Errorf("key not found")
	} else if err != nil {
		return err
	}

	jsonData, ok := data.(string)
	if !ok {
		return fmt.Errorf("data is not a LLMEmbeddingObject")
	} else {
		err = json.Unmarshal([]byte(jsonData), llmEO)
		if err != nil {
			return err
		}
	}

	// Unmarshal JSON data into the LLMEmbeddingObject structure

	return nil
}

// func (llm *LLMContainer) LoadEmbeddData(Index string, options ...LLMCallOption) (map[string]LLMEmbeddingContent, error) {

// }
// Delete removes an embedding object from Redis storage based on a key.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The key used to delete the embedding object from Redis.
//
// Returns:
//   - error: An error if the key cannot be deleted or Redis connection fails.
func (llmEO LLMEmbeddingObject) delete(rdb *redis.Client, KeyID string) error {
	ctx := context.Background()
	// Check Redis connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return err
	}
	// Delete the specified key from Redis
	// _, err = rdb.Del(ctx, KeyID).Result()
	err = deleteKey(ctx, rdb, KeyID, llmEO.EmbeddingPrefix)
	if err != nil {
		return err
	}
	return nil
}

// Save stores an embedding object in Redis after deleting any existing entry.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The key used to store the embedding object in Redis.
//
// Returns:
//   - error: An error if the save operation fails.
func (llmEO *LLMEmbeddingObject) save(rdb *redis.Client, KeyID string) error {
	ctx := context.TODO()
	// Check Redis connection before proceeding
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return err
	}
	err = createIndex(ctx, rdb, llmEO.EmbeddingPrefix)
	if err != nil {
		return err
	}
	// Delete any existing record before saving a new one
	llmEO.delete(rdb, llmEO.getRawDocRedisId())
	// Serialize the embedding object to JSON format
	data, err := json.Marshal(llmEO)
	if err != nil {
		return err
	}
	// Store the serialized data in Redis with no expiration
	err = saveKey(ctx, rdb, KeyID, data)
	if err != nil {
		return fmt.Errorf("error setting JSON in Redis: %v", err)
	}
	return nil
}

// List retrieves multiple embedding objects from Redis with pagination support.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The prefix used to filter stored keys in Redis.
//   - offset: The starting position for retrieval.
//   - limit: The number of items to retrieve.
//
// Returns:
//   - map[string]interface{}: A map containing retrieved objects and total count.
//   - error: An error if the operation fails.
func (llm *LLMContainer) ListEmbeddings(KeyID string, offset, limit int) (map[string]interface{}, error) {
	oe := LLMEmbeddingObject{}
	return oe.list(llm.RedisClient.redisClient, KeyID, offset, limit)
}
func (llmEO LLMEmbeddingObject) list(rdb *redis.Client, KeyID string, offset, limit int) (map[string]interface{}, error) {
	ctx := context.Background()

	// Check Redis connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return nil, err
	}
	// Retrieve all matching keys with the given prefix
	keys, err := rdb.Keys(ctx, KeyID+"*").Result()
	if err != nil {
		return nil, err
	}

	total := len(keys)
	if offset > total {
		offset = total
	}

	end := offset + limit
	if end > total {
		end = total
	}

	// Load embedding objects within the requested range
	var results []LLMEmbeddingObject
	for _, key := range keys[offset:end] {
		var obj LLMEmbeddingObject
		err := obj.load(rdb, key)
		if err == nil {
			results = append(results, obj)
		}
	}

	response := map[string]interface{}{
		"Rows":  results,
		"Total": total,
	}

	return response, nil
}

// saveEmbeddingDataToRedis saves the given embedding object to Redis.
//
// Parameters:
//   - obj: The embedding object to be stored in Redis.
//
// Returns:
//   - error: An error if the save operation fails.
func (llm *LLMContainer) saveEmbeddingDataToRedis(obj LLMEmbeddingObject) error {
	// Store the embedding object in Redis using its generated key
	return obj.save(llm.RedisClient.redisClient, obj.getRawDocRedisId())
}

// RemoveEmbedding deletes an embedding object and its associated keys from Redis.
//
// Parameters:
//   - ObjectId: The unique identifier for the embedding object to be removed.
//   - Index: The Index of the embedding object.
//
// Returns:
//   - error: An error if deletion fails.
func (llm *LLMContainer) RemoveEmbedding(Index string, options ...LLMCallOption) error {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	llmo := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}
	
	// Load the embedding object from Redis
	err:= llmo.load(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
	if err != nil && err.Error() != "key not found" {
		return err
	}


	// Delete all associated keys stored in Redis
	for _, content := range llmo.Contents {
		for _, key := range content.Keys {
			_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
			if err != nil {
				return err
			}
		}
		for _, key := range content.GeneralKeys {
			_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
			if err != nil {
				return err
			}
		}
	}
	//Remove indexes should be implemented
	

	// Remove the embedding object from Redis
	return llmo.delete(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
}

func (llm *LLMContainer) RemoveEmbeddingSubKey(Index, rawDocID string, options ...LLMCallOption) error {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	llmo := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}
	// Load the embedding object from Redis
	llmo.load(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
	keyToDelete := llmo.Contents[rawDocID]
	// Delete all associated keys stored in Redis

	for _, key := range keyToDelete.Keys {
		_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
		if err != nil {
			return err
		}
	}
	for _, key := range keyToDelete.GeneralKeys {
		_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key,false)
		if err != nil {
			return err
		}
	}
	delete(llmo.Contents, rawDocID)
	if len(llmo.Contents) == 0 {
		//deleting the key if it was empty
		return llmo.delete(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
	} else {
		// saving the embedding object to Redis
		return llmo.save(llm.RedisClient.redisClient, llmo.getRawDocRedisId())

	}
}

// GetRagIndexs retrieves the Redis index values for the given documents.
//
// Parameters:
//   - docs: A slice of schema.Document objects containing the documents to search for.
//   - options: Additional options for the search operation.
//
// Returns:
//   - []string: A slice of Redis index values.
//   - error: An error if the operation fails.
func (llm *LLMContainer) GetRagIndexs(docs []schema.Document, options ...LLMCallOption) ([]string, error) {
	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	if len(docs) == 0 {
		return []string{}, nil
	}

	indexName := "rawDocsIdx:"
	if o.Prefix != "" {
		indexName += o.Prefix
	}

	rdb := llm.RedisClient.redisClient
	ctx := context.TODO()

	var escapedQueries []string
	for _, value := range docs {
		escapedValue := escapeRedisQuery(value.Metadata["id"].(string))
		query := fmt.Sprintf(`(@GeneralKeys:{%s}) | (@Keys:{%s})`, escapedValue, escapedValue)
		escapedQueries = append(escapedQueries, query)
	}
	finalQuery := strings.Join(escapedQueries, " | ") // ترکیب همه کوئری‌ها با عملگر `OR`
	results, err := rdb.Do(ctx, "FT.SEARCH", indexName, finalQuery, "RETURN", "1", "$.Index").Result()
	if err != nil {
		return nil, err
	}
	// Extract "index" values from the search results
	var indexValues []string

	// پردازش خروجی FT.SEARCH
	resultsArray, ok := results.(map[interface{}]interface{})
	if !ok || (len(resultsArray) < 2 && len(docs) > 0) {
		// نتایج باید حداقل شامل header و یک نتیجه باشند

		//REDIS COMPATIBILITY
		alternateResultsArray, ok := results.([]interface{})
		if !ok || len(alternateResultsArray) < 2 {
			return nil, fmt.Errorf("no results found")
		} else {
			// resultsArray = alternateResultsArray[1:]
			for _, item := range alternateResultsArray {
				indexData, ok := item.([]interface{})
				if !ok || len(indexData) < 1 {
					continue
				}
				for idx, indexItem := range indexData {
					indexItemData, ok := indexItem.(string)
					if ok && indexItemData == "$.Index" {
						indexValues = append(indexValues, indexData[idx+1].(string))
						break
					}
				}
			}

		}

	} else {
		// پیمایش داده‌های استخراج شده
		for _, item := range resultsArray {
			indexData, ok := item.([]interface{})
			if !ok || len(indexData) < 1 {
				continue
			}
			for _, indexResults := range indexData {
				idxContents, ok := indexResults.(map[interface{}]interface{})
				if ok {
					for _, indexItem := range idxContents {
						indexItemData, ok := indexItem.(string)
						if ok && strings.HasPrefix(indexItemData, "rawDocs:") {
							finalIndex := strings.ReplaceAll(indexItemData, "rawDocs:"+o.Prefix+":", "")
							indexValues = append(indexValues, finalIndex)
							continue
						}
					}
				}
			}
		}

	}
	return indexValues, nil
}

// escapeRedisQuery escapes special characters in a string to be used in Redis queries.
//
// Parameters:
//   - value: The string to be escaped.
//
// Returns:
//   - string: The escaped string.
func escapeRedisQuery(value string) string {
	specialChars := []string{"-", ":"}
	for _, char := range specialChars {
		value = strings.ReplaceAll(value, char, `\`+char)
	}
	return value
}
