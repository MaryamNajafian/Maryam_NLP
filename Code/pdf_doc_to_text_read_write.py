from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This code contains classes needed for reading, editing, and searching through TEXT and PDF files
"""


import PyPDF2
import re

import config


# %%
class ReadTextFile:
    """Contains methods needed for reading TEXT files"""

    def __init__(self, file):
        self.file = file
        self.line = self.file2str()
        self.all_lines = self.file2list

    def __del__(self):
        print("object deleted", self.file)

    def __str__(self):
        return f'{self.all_lines}'

    def __repr__(self):
        return f'{self.file2str()}'

    def file2str(self):
        with open(self.file, 'r') as f:
            return f.readline()

    def file2list(self):
        with open(self.file, 'r') as f:
            return f.readlines()


# %%

class ReadPdfPage:
    """Contains methods needed for reading PDF files"""

    def __init__(self, file, page_number):
        self.file = file
        self.page_number = int(page_number)
        self.page_count = self.total_page_count()
        if 0 <= self.page_number <= self.page_count:
            self.page_content2text = self.pdf2str()
        else:
            self.page_content2text = 'page out of boundary. max page number is ' + str(self.total_page_count())

    def __str__(self):
        return f"content of page {self.page_number}: \n\n {self.page_content2text}"

    def __repr__(self):
        return f"{self.page_content2text}"

    def pdf2str(self):
        f = open(self.file, mode='rb')
        pdf_reader = PyPDF2.PdfFileReader(f)
        self.page_content2text  = pdf_reader.getPage(self.page_number).extractText()
        return self.page_content2text

    def total_page_count(self):
        f = open(self.file, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(f)
        self.page_count = pdf_reader.numPages
        return self.page_count


# %%
class ExtractPdfPage:
    """Contains methods needed for extracting pages from PDF files"""

    def __init__(self, input_file, page_number, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.page_number = page_number

    def page_extracter(self):
        f = open(self.input_file, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(f)
        return pdf_reader.getPage(self.page_number)

    def write_content_to_pdf(self):
        pdf_writer = PyPDF2.PdfFileWriter()
        pdf_writer.addPage(self.page_extracter())
        pdf_output = open(self.output_file, 'wb')
        pdf_writer.write(pdf_output)
        pdf_output.close()

        return True

# %%
class ReadAllPdfPagesOnScreen:
    """Contains methods needed for reading PDF file on the screen"""

    def __init__(self, input_file):
        self.input_file = input_file
        self.page_count = self.total_page_count()

    def read_all_pdf_pages(self):
        pdf_text = list()
        f = open(self.input_file, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(f)
        for p in range(pdf_reader.numPages):
            page = pdf_reader.getPage(p).extractText()
            pdf_text.append(page)
        return pdf_text

    def total_page_count(self):
        f = open(self.input_file, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(f)
        self.page_count = pdf_reader.numPages
        return self.page_count

    def print_content(self):
        pdf_text = self.read_all_pdf_pages()
        for page_number, page_text in enumerate(pdf_text):
            print(f"\n {10 * '.'}{str(page_number)}{10 * '.'} \n \n {page_text}")


# %%
class FindIndexesAndAppearancesOfOccurrences:
    """Contains methods needed for searching strings in the TEXT and returning their position and occurrence"""

    def __init__(self, pattern, text, all_occurrences):
        self.pattern = pattern
        self.text = text
        self.match_count = self.count_number_of_occurrences()
        self.match_appearance_list = self.list_of_pattern_appearances()
        if all_occurrences:
            self.occurrence_list = self.find_start_end_index_for_all_occurrences()
        else:
            self.occurrence_list = self.find_first_start_end_index()
        self.merged_list = self.return_list(self.occurrence_list, self.match_appearance_list, self.match_count)

    def return_list(self, a, b, c):
        merged_list = [(a[i], b[i]) for i in range(c)]
        return merged_list

    def __str__(self):
        return f"occurrence (index, appearance): {self.merged_list}"

    def find_first_start_end_index(self):
        occurrence_index = re.search(self.pattern, self.text)
        return occurrence_index

    def list_of_pattern_appearances(self):
        match_list = re.findall(self.pattern, self.text)
        return match_list

    def count_number_of_occurrences(self):
        match_list = re.findall(self.pattern, self.text)
        return len(match_list)

    def find_start_end_index_for_all_occurrences(self):
        occurrence_index_list = []
        for match in re.finditer(self.pattern, self.text):
            occurrence_index_list.append(match.span())
        return occurrence_index_list


# %%
def main():
    run_application_test = False
    if run_application_test:
        print(help(ReadTextFile))
        text_content = ReadTextFile(file=config.TEXT_DATA_FILE)
        print(text_content)

        pdf_content = ReadPdfPage(file=config.PDF_DATA_FILE, page_number=0)
        print(pdf_content)

        pdf_content = ReadPdfPage(file=config.PDF_DATA_FILE, page_number=10)
        print(pdf_content)

        a = ExtractPdfPage(input_file=config.PDF_DATA_FILE, page_number=0, output_file=config.PDF_DATA_PAGE)
        a.write_content_to_pdf()
        print(help(ExtractPdfPage.write_content_to_pdf))

        a = ReadAllPdfPagesOnScreen(input_file=config.PDF_DATA_FILE)
        a.print_content()

        a = FindIndexesAndAppearancesOfOccurrences(pattern=r'\d{3}-\d{3}-\d{3}',
                                                   text='here are some phone #s 888-999-763 and 929-989-756',
                                                   all_occurrences=True)
        print(a)

        a = FindIndexesAndAppearancesOfOccurrences(pattern=r'(\d{3})-(\d{3})-(\d{3})',
                                                   text='here are some phone #s 888-999-763 and 929-989-756',
                                                   all_occurrences=False)
        print(f"first three digits:{a.occurrence_list.group(1)},"
              f" and last three digits {a.occurrence_list.group(3)} of a phone number")

        a = FindIndexesAndAppearancesOfOccurrences(pattern='is',
                                                   text='The weather is great. Sun is shining. Now isabella is '
                                                        'singing.',
                                                   all_occurrences=True)
        print(a)

        re.findall(r"^\d", "1 plus 2, is 3")  # ^  : starts with
        re.findall(r"\d$", "1 plus 2, is 3")  # $  : ends with
        re.findall(r"[^\d]", "1 plus 2, is 3")  # [^ : excludes
        re.findall(r"[^\d]+", "1 plus 2, is 3")  # + : puts the list of chars back together to form a list of words
        re.findall(r"..at", "the cat sat on the mat with splat")
        re.findall(r"[^.!,:?]+", "she said: this sentence is long. Removed it, ASAP!")  # [^] removes chars inside
        re.findall(r"[\w]+-[\w]", "only find hy-wrd were are the dash-wrds.")  # []+ grab any number of char inside
    pdf_content = ReadPdfPage(file=config.PDF_DATA_FILE, page_number=0)
    print(pdf_content)

main()
# %%
if __name__ == "__main__":
    print(__doc__)
    main()
