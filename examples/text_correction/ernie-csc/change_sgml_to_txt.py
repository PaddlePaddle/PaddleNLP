from html.parser import HTMLParser
import re
import argparse
import xml.dom.minidom
from xml.dom.minidom import parse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", default="train.sgml", type=str)
parser.add_argument("--output", "-o", default="train.txt", type=str)

args = parser.parse_args()


def main():
    with open(args.output, "w", encoding="utf-8") as fw:
        dom = xml.dom.minidom.parse(args.input)
        example_nodes = dom.documentElement.getElementsByTagName('SENTENCE')
        for example in example_nodes:
            raw_text = example.getElementsByTagName('TEXT')[0].childNodes[
                0].data
            correct_text = list(raw_text)
            mistakes = example.getElementsByTagName('MISTAKE')
            for mistake in mistakes:
                loc = int(
                    mistake.getElementsByTagName('LOCATION')[0].childNodes[0]
                    .data) - 1
                correction = mistake.getElementsByTagName('CORRECTION')[
                    0].childNodes[0].data
                correct_text[loc] = correction

            correct_text = ''.join(correct_text)
            fw.write("{}\t{}\n".format(raw_text, correct_text))


if __name__ == "__main__":
    main()
