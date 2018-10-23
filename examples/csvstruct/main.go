// Example decoder for tab separated data. CSV looks like a simple format, but
// its surprisingly hard to create valid CSV files. Sometimes simply splitting
// a string circumvents all the problems that come with quoting styles, field
// counts and so on.
package main

import (
	"bufio"
	"io"
	"log"
	"os"
	"reflect"
	"sync"

	"strings"

	"github.com/fatih/structs"
)

type Record struct {
	Name  string `csv:"name"`
	Plate string `csv:"plate"`
}

func main() {
	dec := NewDecoderSeparator(os.Stdin, ",")
	for {
		var record Record
		err := dec.Decode(&record)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%+v", record)
	}
}

// A Decoder reads and decodes rows from an input stream.
type Decoder struct {
	Header    []string    // Column names.
	Separator string      // Field separator.
	r         *SkipReader // The underlying reader.
	once      sync.Once
}

// NewDecoder returns a new decoder with tab as field separator.
func NewDecoder(r io.Reader) *Decoder {
	return NewDecoderSeparator(r, "\t")
}

// NewDecoderSeparator creates a new decoder with a given separator.
func NewDecoderSeparator(r io.Reader, sep string) *Decoder {
	return &Decoder{r: NewSkipReader(bufio.NewReader(r)), Separator: sep}
}

// readHeader attempts to read the first row and store the column names. If the
// header has been already set manually, the values won't be overwritten.
func (dec *Decoder) readHeader() (err error) {
	dec.once.Do(func() {
		if len(dec.Header) > 0 {
			return
		}
		var line string
		if line, err = dec.r.ReadString('\n'); err != nil {
			return
		}
		dec.Header = strings.Split(line, dec.Separator)
	})
	return
}

// Decode a single entry, reuse csv struct tags.
func (dec *Decoder) Decode(v interface{}) error {
	if err := dec.readHeader(); err != nil {
		return err
	}
	if reflect.TypeOf(v).Elem().Kind() != reflect.Struct {
		return nil
	}
	line, err := dec.r.ReadString('\n')
	if err == io.EOF {
		return io.EOF
	}
	record := strings.Split(line, dec.Separator)

	s := structs.New(v)

	for _, f := range s.Fields() {
		tag := f.Tag("csv")
		if tag == "" || tag == "-" {
			continue
		}
		for i, header := range dec.Header {
			if i >= len(record) {
				break // Record has too few columns.
			}
			if tag != header {
				continue
			}
			if err := f.Set(record[i]); err != nil {
				return err
			}
		}
	}
	return nil
}

// SkipReader skips empty lines and lines with comments.
type SkipReader struct {
	r               *bufio.Reader
	CommentPrefixes []string
}

// NewSkipReader creates a new SkipReader.
func NewSkipReader(r *bufio.Reader) *SkipReader {
	return &SkipReader{r: r}
}

// ReadString will return only non-empty lines and lines not starting with a comment prefix.
func (r SkipReader) ReadString(delim byte) (s string, err error) {
	for {
		s, err = r.r.ReadString(delim)
		if err == io.EOF {
			return
		}
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		for _, p := range r.CommentPrefixes {
			if strings.HasPrefix(s, p) {
				continue
			}
		}
		break
	}
	return
}
