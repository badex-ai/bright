'use client'; // This directive is needed for client-side components in Next.js App Router

import { useState } from 'react';

interface ProductData {
  name: string;
  price: string;
  currency: string;
  availability: string;
  source_url: string;
}

interface SearchResponse {
  products: ProductData[];
  analysis_chart_base64?: string;
  analysis_summary?: string;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/search', { // Assuming FastAPI runs on 8000
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Something went wrong with the search.');
      }

      const data: SearchResponse = await response.json();
      console.log('Search results:', data);
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-2xl">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">
          Luxury Item Search & Analysis
        </h1>
        <form onSubmit={handleSearch} className="flex flex-col gap-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Rolex Submariner, Hermes Birkin bag"
            className="p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />
          <button

            type="submit"
            className="bg-blue-600 text-white p-3 rounded-md hover:bg-blue-700 transition-colors disabled:bg-blue-400"
            disabled={loading}
          >
            {loading ? 'Searching...' : 'Search Luxury Item'}
          </button>
        </form>

        {error && (
          <p className="mt-4 text-red-600 text-center">{error}</p>
        )}

        {results && (
          <div className="mt-8 border-t pt-8 border-gray-200">
            {results.products.length > 0 ? (
              <>
                <h2 className="text-2xl font-semibold text-gray-700 mb-4">
                  Search Results
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.products.map((product, index) => (
                    <div key={index} className="border p-4 rounded-md bg-gray-50">
                      <h3 className="text-lg font-medium text-gray-800">{product.name}</h3>
                      <p className="text-gray-600">Price: {product.price} {product.currency}</p>
                      <p className="text-gray-600">Availability: {product.availability}</p>
                      <a
                        href={product.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-500 hover:underline text-sm mt-2 block"
                      >
                        View Source
                      </a>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-gray-600 text-center">No products found for your query.</p>
            )}

          

         
          </div>
        )}
      </div>
    </div>
  );
}

