import os
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd


class Crawler:
    def __init__(self, url=None, params=None, headers=None):
        self.url = "https://www.etsy.com/market/ai_art?" if url is None else url
        self.params = {"ref": "pagination", "page": 1} if params is None else params
        self.headers = (
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                "Connection": "keep-alive",
            }
            if headers is None
            else headers
        )
        self.negative_keywords = [
            "prompt",
            "product",
            "template",
            "commission",
            "00",
            "idea",
            "adult",
            "nude",
            "x-rated",
            "nsfw",
            "naked",
            "mature",
        ]
        self.negative_sellers = [
            "OnlyNSFWHub",
            "AIArtCenter",
            "XartPhotography",
            "MyAIbeauties",
        ]
        self.titles = []
        self.meta_df = None

    def _get_content(self, url=None, params=None):
        url = self.url if url is None else url
        response = requests.get(url, params=params, headers=self.headers)
        time.sleep(1)
        if response.status_code != 200:
            print(response.content)
            print(response.status_code)
            print(f"Failed to load page {url}")
            return None

        return response.content

    def set_negatives(
        self, negative_keywords: list = None, negative_sellers: list = None
    ):
        self.negative_keywords += (
            negative_keywords if negative_keywords is not None else []
        )
        self.negative_sellers += (
            negative_sellers if negative_sellers is not None else []
        )

    def _get_meta(self, item):
        title = item.find(class_="v2-listing-card__title").get_text(strip=True)
        if title in self.titles:
            return None

        if (
            any(neg in title.lower() for neg in self.negative_keywords)
            or title in self.titles
        ):
            return None

        img_src = item.find("img")
        rating = item.find("input", {"name": "rating"})
        num_reviews = item.find("span", class_="wt-text-body-01")
        price = item.find("span", class_="currency-value")

        # get detail informations (seller, description)
        detail_url = item.find("a", class_="listing-link")

        if None in [title, img_src, rating, num_reviews, price, detail_url]:
            return None
        else:
            img_src = img_src["src"]
            rating = rating["value"]
            num_reviews = num_reviews.get_text(strip=True)
            price = price.get_text(strip=True)
            detail_url = detail_url["href"]

        return {
            "title": title,
            "price": price,
            "rating": rating,
            "num_reviews": num_reviews,
            "detail_url": detail_url,
            "img_src": img_src,
        }

    def crawl(self, content: str, meta_path=None):
        meta_path = "metadata" if meta_path is None else meta_path

        if not os.path.exists(meta_path):
            os.makedirs(meta_path)

        columns = [
            "id",
            "title",
            "price",
            # "seller",
            "rating",
            "num_reviews",
            "detail_url",
            "img_src",
            # "num_seller_img",
            # "num_review_img",
            # "reviews",
            # "description"
        ]

        if os.path.exists(f"{meta_path}/metadata.csv"):
            self.meta_df = pd.read_csv(f"{meta_path}/metadata.csv")
            self.titles = self.meta_df["title"].tolist()
        else:
            self.meta_df = pd.DataFrame(columns=columns)
            self.titles = []

        soup = BeautifulSoup(content, "html.parser")
        items = soup.find_all("div", class_="v2-listing-card")

        for item in items:
            meta = self._get_meta(item)

            if meta is None:
                continue

            self.titles.append(meta["title"])
            meta["id"] = str(len(self.meta_df) + 1)

            new_row = pd.DataFrame([meta], columns=columns)
            self.meta_df = pd.concat([self.meta_df, new_row])

        self.meta_df.to_csv(f"{meta_path}/metadata.csv", index=False)

        return self.meta_df

    def save_images(self, img_path: str = None):
        img_path = "images"
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        if self.meta_df is None:
            raise ValueError("Metadata not found.")

        img_src_list = list(self.meta_df["img_src"])

        for i, img_src in enumerate(img_src_list):
            img = requests.get(img_src)
            with open(f"{img_path}/{i+1}.jpg", "wb") as f:
                f.write(img.content)

    def load_meta(self, meta_path=None):
        meta_path = "metadata/metadata.csv" if meta_path is None else meta_path

        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata not found in {meta_path}")

        self.meta_df = pd.read_csv(f"{meta_path}")

        return self

    def get_details(self):
        detail_df = pd.DataFrame(
            columns=[
                "id",
                "seller",
                "num_seller_img",
                "num_review_img",
                "description",
            ]
        )
        for _, data in self.meta_df.iterrows():
            detail_url = data["detail_url"]

            content = self._get_content(detail_url)

            if content is None:
                return None

            detail_soup = BeautifulSoup(content, "html.parser")

            seller = detail_soup.find(
                "a", class_="wt-text-link-no-underline wt-text-black"
            )
            description = detail_soup.find("div", {"data-id": "description-text"})
            seller_img_list = detail_soup.findAll(
                "li", class_="carousel-pagination-item-v2"
            )
            review_img_list = detail_soup.find(
                "div", attrs={"data-appears-component-name": "customer_photos"}
            )

            if seller is None or description is None:
                seller = "None"
                description = "None"
                num_review_img = 0
                num_seller_img = 0
            else:
                seller = seller.get_text(strip=True)
                description = description.find("p").get_text(strip=True)

                num_seller_img = len(seller_img_list) if seller_img_list else 0
                num_review_img = (
                    len(review_img_list.find_all("img")) if review_img_list else 0
                )

            if any(neg in seller for neg in self.negative_sellers):
                return None

            new_row = pd.DataFrame(
                [
                    {
                        "id": data["id"],
                        "seller": seller,
                        "num_seller_img": num_seller_img,
                        "num_review_img": num_review_img,
                        "description": description,
                    }
                ]
            )
            detail_df = pd.concat([detail_df, new_row])

        self.meta_df = pd.merge(self.meta_df, detail_df, on="id")

        return self.meta_df
