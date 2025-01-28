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
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gabriel-vasile/mimetype"
	"github.com/google/go-tika/tika"
	"github.com/ledongthuc/pdf"
)

// Transcriber handles document transcription by extracting text from various file formats.
//
// This struct provides configurations and options for processing documents through the Tika service,
// managing temporary files, and setting processing limits.
//
// Fields:
//   - MaxPageLimit: The maximum number of pages to process in a document (JUST PDF Documents).
//   - TikaURL: The URL of the Apache Tika server used for text extraction.
//   - initialized: A boolean indicating if the transcriber has been initialized successfully.
//   - TempFolder: The folder where temporary files will be stored during processing (Downloading / Transcribing).
//   - folderSep: The file path separator used for compatibility across operating systems.
type Transcriber struct {
	MaxPageLimit uint   // Maximum number of pages allowed for processing
	TikaURL      string // URL of the Apache Tika service for text extraction
	initialized  bool   // Indicates if the transcriber is initialized
	TempFolder   string // Path to the temporary folder for storing transcribed files
	folderSep    string // File separator ("/" for Linux, "\" for Windows)
}

// TranscribeConfig provides configuration settings for document transcription.
//
// This struct specifies options for text extraction, including OCR settings,
// language preferences, and processing time limits.
//
// Fields:
//   - TikaLanguage: The language code used by Tika OCR for text extraction.
//   - Language: The target language for transcription results.
//   - OCROnly: A flag to indicate whether to perform only Optical Character Recognition (OCR).
//   - ExtractInlineImages: A flag to extract text from inline images within the document.
//   - MaxTimeout: The maximum allowed duration for document processing.

type TranscribeConfig struct {
	TikaLanguage        string        //PDF ONLY, OCR language code (refer to Tesseract OCR languages) can be found @ https://github.com/tesseract-ocr/tessdata/
	Language            string        // Target language for transcription
	OCROnly             bool          // Perform OCR only, ignoring non-image text
	ExtractInlineImages bool          // Enable extraction of text from inline images
	MaxTimeout          time.Duration // Maximum processing time before timeout
}

// init initializes the Transcriber instance by setting default values and preparing the environment.
//
// This function ensures the transcriber is properly set up with default page limits,
// Tika URL configurations, and temporary folder paths.
//
// Returns:
//   - error: An error if the initialization fails.
func (Ts *Transcriber) init() error {
	if !Ts.initialized {
		if Ts.MaxPageLimit == 0 {
			Ts.MaxPageLimit = 20   // Default page limit if not specified
		}
		
		Ts.initialized = true
		Ts.folderSep = "/"
		if runtime.GOOS == "windows" {
			Ts.folderSep = "\\"
		}
		if Ts.TempFolder == "" {
			exePath, err := os.Executable()
			if err != nil {
				fmt.Println("error fetching application folder:", err)
				return err
			}
			Ts.TempFolder = filepath.Dir(exePath) + Ts.folderSep + "tmp"
			 

		}
	}
	return nil
}

// transcribeURL downloads and processes content from a given URL.
//
// This function downloads the content from the URL, detects the MIME type, and extracts text 
// based on file type (PDF, HTML, etc.).
//
// Parameters:
//   - inputURL: The URL of the document to be transcribed.
//   - tc: Transcription configuration settings.
//
// Returns:
//   - string: Extracted text content.
//   - int: Number of pages processed (if applicable).
//   - error: An error if the transcription fails.
func (Ts *Transcriber) transcribeURL(inputURL string, tc TranscribeConfig) (string, int, error) {
	Ts.init()
	log.Println("Downloading " + inputURL + "...")
	fileContents, mimeType, fileName, _, fetchErr := Ts.downloadPage(inputURL)
	if fetchErr != nil {
		return "", 0, fetchErr
	}
	switch {
	case strings.Contains(mimeType, "application/pdf"):
		return Ts.getPDFContents(tc, fileName)
	case strings.Contains(mimeType, "text/html"):
		extractedInfo := Ts.extractHTMLContent(fileContents)
		return extractedInfo, 0, nil
	default:
		return "", 0, fmt.Errorf("file type not supported")
	}
}

// transcribeFile processes a local file and extracts text based on its MIME type.
//
// The function detects the MIME type if not provided, and processes the file accordingly 
// (e.g., using OCR for PDFs or parsing plain text files).
//
// Parameters:
//   - fileName: The path to the file to be transcribed.
//   - mimeType: The MIME type of the file (if known, otherwise it will be detected).
//   - tc: Transcription configuration settings.
//
// Returns:
//   - string: Extracted text content.
//   - int: Number of pages processed (if applicable).
//   - error: An error if the transcription fails.
func (Ts *Transcriber) transcribeFile(fileName, mimeType string, tc TranscribeConfig) (string, int, error) {
	Ts.init()
	if mimeType == "" {
		detectedMimeType, mimedetectionErr := mimetype.DetectFile(fileName)
		if mimedetectionErr != nil {
			mimeType = "application/pdf"
		} else {

			mimeType = detectedMimeType.String()
		}
	}
	switch {
	case strings.Contains(mimeType, "application/pdf"):
		return Ts.getPDFContents(tc, fileName)
	case strings.Contains(mimeType, "text/html"):
		fileContents, err := os.ReadFile(fileName)
		if err != nil {
			return "", 0, err
		}
		extractedInfo := Ts.extractHTMLContent(fileContents)
		return extractedInfo, 0, nil
	case strings.Contains(mimeType, "text/plain"):
		fileContents, err := os.ReadFile(fileName)
		if err != nil {
			return "", 0, err
		}
		extractedInfo := Ts.extractTextContent(fileContents)
		return extractedInfo, 0, nil
	default:
		return Ts.getContentsFromTika(tc, fileName)

	}

}

// downloadPage downloads the content from a given URL and caches it locally if not already cached.
//
// The function checks for a cached version of the file and downloads it if necessary, 
// saving it to the temporary folder.
//
// Parameters:
//   - urlToGet: The URL of the page to download.
//
// Returns:
//   - []byte: The downloaded content as byte data.
//   - string: The MIME type of the content.
//   - string: The local file path where the content is stored.
//   - bool: Whether the content was retrieved from the cache.
//   - error: An error if the download fails.
func (Ts *Transcriber) downloadPage(urlToGet string) ([]byte, string, string, bool, error) {
	cached := false
	var result []byte
	var err error
	mimeType := ""
	fileName := Ts.prepareFileName(urlToGet)

	_, urlParseErr := url.Parse(urlToGet)
	if urlParseErr != nil {
		return result, mimeType, fileName, cached, urlParseErr
	}

	destinationFolder := Ts.TempFolder + Ts.folderSep + time.Now().Format("2006-01-02")
	filePath := destinationFolder + Ts.folderSep + fileName
	result, err = os.ReadFile(filePath)
	if err == nil {
		cached = true
		mimeTypeBytes, _ := os.ReadFile(filePath + ".meta")
		mimeType := string(mimeTypeBytes)
		return result, mimeType, filePath, cached, nil
	} else {
		// cmslog.Log(" downloading "+urlToGet, "", 80)
		result, mimeType, downloadErr := Ts.downloadRemoteFileWithMimeType(urlToGet)
		if downloadErr != nil {
			return result, mimeType, filePath, cached, downloadErr
		}

		_, err = os.Stat(destinationFolder)
		if os.IsNotExist(err) {
			err = os.MkdirAll(destinationFolder, os.ModePerm)
			if err != nil {
				return result, mimeType, filePath, cached, errors.New("error creating temp folder")
			}
		}
		err := os.WriteFile(filePath, result, 0666)
		if err != nil {
			return result, mimeType, filePath, cached, err
		} else {
			_ = os.WriteFile(filePath+".meta", []byte(mimeType), 0666)
			return result, mimeType, filePath, cached, nil
		}
	}
}
// getContentsFromTika extracts text from a document using Apache Tika.
//
// This function sends the document to a Tika server for text extraction and handles OCR settings.
//
// Parameters:
//   - tc: Transcription configuration settings.
//   - inputPath: The path to the file to be processed.
//
// Returns:
//   - string: Extracted text content.
//   - int: Number of pages processed.
//   - error: An error if extraction fails.

func (Ts *Transcriber) getContentsFromTika(tc TranscribeConfig, inputPath string) (string, int, error) {
	f, err := os.Open(inputPath)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	client := tika.NewClient(nil, Ts.TikaURL)
	pageCount := -1

	header := http.Header{"Accept": []string{"text/plain"}}
	//
	if tc.Language != "" {
		header.Add("X-Tika-OCRLanguage", tc.TikaLanguage)
	}
	if tc.OCROnly {
		header.Add("X-Tika-PDFOcrStrategy", "ocr_only")
	}
	if tc.ExtractInlineImages {
		header.Add("X-Tika-PDFextractInlineImages", "true")
	}

	timeout := int64(tc.MaxTimeout) / 1000000
	if timeout == 0 {
		tc.MaxTimeout = 1 * time.Minute
		timeout = int64(tc.MaxTimeout) / 1000000
	}
	header.Add("X-Tika-Timeout-Millis", fmt.Sprintf("%d", timeout))

	ioReadCloser, err := client.ParseReaderWithHeader(context.Background(), f, header)
	if err != nil {
		return "", pageCount, err
	}
	buf := new(strings.Builder)
	io.Copy(buf, ioReadCloser)
	result := buf.String()
	result = Ts.cleanupText(result)
	return result, pageCount, nil
}

// getPDFContents extracts text content from a PDF file.
//
// This function first checks the number of pages in the PDF and then processes it using
// the Apache Tika service for OCR-based text extraction if necessary.
//
// Parameters:
//   - tc: Transcription configuration settings (e.g., OCR options, language settings).
//   - inputPath: The file path of the PDF document to be processed.
//
// Returns:
//   - string: The extracted text content from the PDF document.
//   - int: The number of pages in the document.
//   - error: An error if the file cannot be processed.

func (Ts *Transcriber) getPDFContents(tc TranscribeConfig, inputPath string) (string, int, error) {
	result := ""

	pageCount := -1

	_, r, err := pdf.Open(inputPath)
	if err != nil {
		return "", 0, err
	}
	pageCount = r.NumPage()
	if pageCount > int(Ts.MaxPageLimit) {

		return "", pageCount, errors.New("PDF file has more than " + fmt.Sprintf("%d", Ts.MaxPageLimit) + " pages")
	}

	result, pageCount, err = Ts.getContentsFromTika(tc, inputPath)
	return result, pageCount, err

}

/*** Tools ***/

// cleanupText removes unnecessary whitespace, special characters, and formatting inconsistencies
// from the extracted text content.
//
// This function performs cleanup operations such as removing extra line breaks,
// multiple dashes, and unnecessary spaces to ensure clean output.
//
// Parameters:
//   - textContent: The extracted raw text content.
//
// Returns:
//   - string: The cleaned-up text content.
func (Ts *Transcriber) cleanupText(textContent string) string {
	textContent = strings.ReplaceAll(textContent, "\t", "")
	hasEnter := true
	for hasEnter {
		textContent = strings.ReplaceAll(textContent, "\n\n", "\n")
		hasEnter = strings.Contains(textContent, "\n\n")
	}

	hasMultipleDash := true
	for hasMultipleDash {
		textContent = strings.ReplaceAll(textContent, "----", "")
		hasMultipleDash = strings.Contains(textContent, "----")
	}

	hasSpaceEnter := true
	for hasSpaceEnter {
		textContent = strings.ReplaceAll(textContent, "\n \n", "\n")
		hasSpaceEnter = strings.Contains(textContent, "\n \n")
	}
	hasSpaceEnter = true
	for hasSpaceEnter {
		textContent = strings.ReplaceAll(textContent, "\n \n", "\n")
		hasSpaceEnter = strings.Contains(textContent, "\n \n")
	}
	return textContent
}

// prepareFileName sanitizes a URL to generate a valid and unique filename.
//
// This function replaces special characters in the URL with underscores to ensure the resulting
// filename is safe for storage and retrieval purposes.
//
// Parameters:
//   - urlToGet: The original URL to be sanitized.
//
// Returns:
//   - string: A sanitized version of the URL suitable for use as a filename.
func (Ts Transcriber) prepareFileName(urlToGet string) string {
	fileName := strings.ReplaceAll(urlToGet, ".", "_")
	fileName = strings.ReplaceAll(fileName, ":", "_")
	fileName = strings.ReplaceAll(fileName, "/", "_")
	fileName = strings.ReplaceAll(fileName, "&", "_")
	fileName = strings.ReplaceAll(fileName, "?", "_")
	fileName = strings.ReplaceAll(fileName, ">", "_")
	fileName = strings.ReplaceAll(fileName, "<", "_")
	fileName = strings.ReplaceAll(fileName, "!", "_")
	fileName = strings.ReplaceAll(fileName, "#", "_")
	return fileName
}
// extractHTMLContent extracts readable text from HTML content.
//
// This function parses HTML content to extract text from headings, paragraphs, and tables,
// discarding unnecessary HTML tags and metadata.
//
// Parameters:
//   - htmlBytes: The raw HTML content as a byte slice.
//
// Returns:
//   - string: Extracted text content formatted for readability.
func (Ts Transcriber) extractHTMLContent(htmlBytes []byte) string {
	// Create a reader from the byte slice
	reader := bytes.NewReader(htmlBytes)

	// Parse the HTML
	doc, err := goquery.NewDocumentFromReader(reader)
	if err != nil {
		return ""
	}

	var output strings.Builder

	// Extract title
	title := doc.Find("title").First().Text()
	if title != "" {
		output.WriteString(strings.TrimSpace(title) + "\n")
	}

	// Extract text from headings and paragraphs
	doc.Find("h1, h2, h3, h4, h5, h6, p").Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())
		if text != "" {
			output.WriteString(text + "\n")
		}
	})

	// Extract tables
	doc.Find("table").Each(func(i int, table *goquery.Selection) {
		output.WriteString("[Table Start]\n")

		// Extract headers
		var headers []string
		table.Find("th").Each(func(j int, header *goquery.Selection) {
			headers = append(headers, strings.TrimSpace(header.Text()))
		})
		output.WriteString(strings.Join(headers, " | ") + " | \n")

		// Extract rows
		table.Find("tr").Each(func(j int, row *goquery.Selection) {
			var rowData []string
			row.Find("td").Each(func(k int, cell *goquery.Selection) {
				rowData = append(rowData, strings.TrimSpace(cell.Text()))
			})

			if len(rowData) > 0 {
				output.WriteString(strings.Join(rowData, " | ") + " | \n")
			}
		})

		output.WriteString("[Table End]\n")
	})

	return output.String()
}

// extractTextContent processes raw text content to remove unwanted characters and whitespace.
//
// This function cleans up plain text content to ensure consistency and readability.
//
// Parameters:
//   - fileBytes: The raw text content as a byte slice.
//
// Returns:
//   - string: Cleaned-up text content.
func (Ts Transcriber) extractTextContent(fileBytes []byte) string {
	// Create a reader from the byte slice
	return Ts.cleanupText(string(fileBytes))
}

// downloadRemoteFileWithMimeType downloads a file from the provided URL and determines its MIME type.
//
// This function makes an HTTP GET request to fetch the file contents, detects the MIME type from
// the response headers, and returns the file's contents along with the MIME type.
//
// Parameters:
//   - urlToGet: The URL of the file to download.
//
// Returns:
//   - []byte: The content of the downloaded file.
//   - string: The detected MIME type of the file.
//   - error: An error if the download or MIME detection fails.
func (Ts Transcriber) downloadRemoteFileWithMimeType(urlToGet string) ([]byte, string, error) {
	client := &http.Client{}
	mimeType := ""
	req, err := http.NewRequest("GET", urlToGet, nil)
	if err != nil {
		return nil, mimeType, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := client.Do(req)
	if err != nil {
		return nil, mimeType, err
	}
	mimeType = resp.Header.Get("Content-Type")
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		body, _ := io.ReadAll(resp.Body)
		return body, mimeType, nil
	} else {
		body, _ := io.ReadAll(resp.Body)
		return body, mimeType, errors.New("http status error")
	}

}
