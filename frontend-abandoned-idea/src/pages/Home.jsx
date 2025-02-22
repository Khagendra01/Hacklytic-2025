import { Link } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';

export default function Home() {
  const { user } = useSelector(state => state.auth);
  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 opacity-90"></div>
      <div className="relative max-w-7xl mx-auto py-24 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl md:text-6xl">
            Welcome to MicroBet
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-200">
            Your premier destination for online betting. Place bets, track your progress, and win big!
          </p>
          <div className="mt-10 flex justify-center space-x-4">
            {user? <>Hello, {user.name}</> : <>
            <Link
              to="/register"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Get Started
            </Link>
            <Link
              to="/login"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-white hover:bg-gray-50"
            >
              Sign In
            </Link>
            </> }
          </div>
        </div>
      </div>
    </div>
  );
}