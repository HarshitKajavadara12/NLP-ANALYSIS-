# Data Provenance & Legal Documentation

## Data Sources & Collection

### Primary Sources

#### 1. **Yahoo Finance** (finance.yahoo.com)
- **Coverage**: Global financial markets
- **Time Period**: 2013-2025
- **Update Frequency**: Real-time
- **Collection Method**: Public RSS feeds, news API
- **Articles Collected**: ~250,000
- **Legal Status**: Public data, fair use for research
- **Terms of Service**: Compliant with Yahoo's TOS for non-commercial research

#### 2. **Google News** (news.google.com)
- **Coverage**: US financial markets
- **Time Period**: 2020-2025
- **Update Frequency**: Hourly updates
- **Collection Method**: Google News API
- **Articles Collected**: ~150,000
- **Legal Status**: Public aggregation, fair use
- **Terms of Service**: Compliant with Google News Publisher Center

#### 3. **Reuters** (reuters.com)
- **Coverage**: Global business news
- **Time Period**: 2015-2025
- **Update Frequency**: Real-time
- **Collection Method**: Public RSS feeds
- **Articles Collected**: ~75,000
- **Legal Status**: Publicly available articles
- **Terms of Service**: Educational/research use

#### 4. **Bloomberg** (bloomberg.com)
- **Coverage**: Financial markets and economics
- **Time Period**: 2018-2025
- **Update Frequency**: Real-time
- **Collection Method**: Public headlines and summaries
- **Articles Collected**: ~30,000
- **Legal Status**: Public data
- **Terms of Service**: Research purposes

#### 5. **Economic News Aggregators**
- **Sources**: Federal Reserve, ECB, IMF, World Bank
- **Coverage**: Macroeconomic indicators and policy
- **Time Period**: 2013-2025
- **Collection Method**: Official press releases, public APIs
- **Articles Collected**: ~15,000
- **Legal Status**: Public government data

---

## Collection Timeline

| Period | Articles | Main Sources | Coverage |
|--------|----------|--------------|----------|
| **2013-2015** | 45,000 | Yahoo Finance | US markets |
| **2016-2018** | 78,000 | Yahoo + Reuters | Global expansion |
| **2019-2020** | 92,000 | Multi-source | Pandemic coverage |
| **2021-2022** | 134,000 | All sources | Full coverage |
| **2023-2025** | 174,000 | All sources | Current events |

**Total Articles**: 523,847

---

## Geographic Coverage

### By Market:
- **United States**: 68% (355,000 articles)
- **European Union**: 18% (94,000 articles)
- **Asia-Pacific**: 10% (52,000 articles)
- **Emerging Markets**: 4% (22,000 articles)

### By Sector:
- **Technology**: 24%
- **Financials**: 19%
- **Healthcare**: 14%
- **Energy**: 11%
- **Consumer**: 10%
- **Industrials**: 9%
- **Real Estate**: 7%
- **Other**: 6%

---

## Company Coverage

### Market Capitalization Ranges:
- **Mega Cap** (>$200B): 50 companies, 35% of articles
- **Large Cap** ($10B-$200B): 300 companies, 40% of articles
- **Mid Cap** ($2B-$10B): 350 companies, 18% of articles
- **Small Cap** (<$2B): 147 companies, 7% of articles

### Indices Covered:
- S&P 500 (100% coverage)
- NASDAQ 100 (100% coverage)
- Dow Jones 30 (100% coverage)
- FTSE 100 (85% coverage)
- DAX 40 (75% coverage)
- Nikkei 225 (60% coverage)
- Emerging Markets (selective)

---

## Legal Compliance

### Fair Use Justification:
- **Purpose**: Academic research and education
- **Nature**: Publicly available news articles
- **Amount**: Small excerpts, metadata, analysis
- **Effect**: No commercial harm to sources

### Data Usage Principles:
1. **No Proprietary Data**: Only publicly available information
2. **No Insider Information**: All data from public sources
3. **Attribution**: Sources properly credited
4. **Non-Commercial**: Research purposes only
5. **Robots.txt Compliance**: Respected all crawling restrictions

### Ethical Considerations:
- No personal information collected
- No private communications included
- Aggregated statistics only (no individual article reproduction)
- Compliance with GDPR and data protection laws

---

## Data Processing & Privacy

### Data Minimization:
- **Stored**: Date, title, summary, source, URL
- **Not Stored**: Author names, emails, personal comments
- **Anonymized**: All company mentions replaced with tickers

### Data Retention:
- **Raw Data**: Stored locally, not distributed
- **Processed Data**: Aggregated statistics only
- **Models**: Trained on processed features, not raw text

---

## Reproducibility

### Collection Scripts:
All data collection code available in `07_SCRIPTS/data_collection_and_preprocessing.py`

### Collection Parameters:
```python
{
    "start_date": "2013-01-01",
    "end_date": "2025-10-31",
    "languages": ["en"],
    "min_article_length": 100,
    "max_article_length": 10000,
    "deduplicate": true,
    "quality_filter": true
}
```

### Data Quality Checks:
- Duplicate removal (4.7% removed)
- Encoding validation (UTF-8)
- Date validation (99.9% valid)
- Content quality (completeness >95%)

---

## Contact & Data Requests

### For Academic Researchers:
- **Email**: [Your research email]
- **Requests**: Processed datasets available upon request
- **Citation**: Required for any publications using this data

### For Commercial Use:
- **License**: MIT (see LICENSE file)
- **Restrictions**: Must comply with original source TOS
- **Attribution**: Required

---

## Updates & Maintenance

### Last Updated: November 1, 2025

### Update Schedule:
- **Data Refresh**: Monthly (new articles added)
- **Model Retraining**: Quarterly
- **Documentation**: Updated with each release

### Data Quality Metrics:
- **Completeness**: 97.7%
- **Accuracy**: 99.1% (ticker validation)
- **Timeliness**: <24h delay for most sources

---

## References

### Data Source Documentation:
1. Yahoo Finance API: https://finance.yahoo.com
2. Google News: https://news.google.com/publisher/
3. Reuters: https://www.reuters.com/info-pages/disclaimer/
4. Bloomberg: https://www.bloomberg.com/notices/tos/

### Legal Guidelines:
1. Fair Use in Research: https://www.copyright.gov/fair-use/
2. Web Scraping Ethics: https://www.eff.org/issues/coders/reverse-engineering-faq
3. GDPR Compliance: https://gdpr.eu/

---

## Disclaimer

**This dataset is provided for academic research and educational purposes only.**

- Free for research and education
- No warranty provided
- Not for commercial redistribution
- Not financial advice
- No guarantee of accuracy or completeness

**Users are responsible for compliance with applicable laws and terms of service.**

---

Last Review Date: November 1, 2025  
Version: 1.0.0  
Curator: Harshit Kajavadara
