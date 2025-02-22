import { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setBets, addBet } from '../store/slices/betsSlice';
import { updateBalance } from '../store/slices/authSlice';

export default function Dashboard() {
  const dispatch = useDispatch();
  const { bets, availableBets } = useSelector(state => state.bets);
  const { user } = useSelector(state => state.auth);
  const [selectedBet, setSelectedBet] = useState(null);
  const [betAmount, setBetAmount] = useState('');
  const [selectedOutcome, setSelectedOutcome] = useState('');

  useEffect(() => {
    // Simulate fetching bets from API
    const fetchBets = async () => {
      const dummyBets = [
        { id: 1, type: 'Football', amount: 100, odds: 1.8, status: 'Won', date: '2024-03-10' },
        { id: 2, type: 'Basketball', amount: 50, odds: 2.1, status: 'Lost', date: '2024-03-09' },
        { id: 3, type: 'Tennis', amount: 75, odds: 1.5, status: 'Pending', date: '2024-03-11' },
      ];
      dispatch(setBets(dummyBets));
    };

    fetchBets();
  }, [dispatch]);

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'won':
        return 'text-green-600';
      case 'lost':
        return 'text-red-600';
      default:
        return 'text-yellow-600';
    }
  };

  const handlePlaceBet = () => {
    if (!selectedBet || !betAmount || !selectedOutcome) return;

    const amount = parseFloat(betAmount);
    if (amount <= 0 || amount > user.balance) return;

    const newBet = {
      id: Date.now(),
      type: selectedBet.type,
      event: selectedBet.event,
      amount: amount,
      odds: selectedBet.odds[selectedOutcome],
      status: 'Pending',
      date: new Date().toISOString().split('T')[0],
      outcome: selectedOutcome
    };

    dispatch(addBet(newBet));
    dispatch(updateBalance(user.balance - amount));
    
    // Reset form
    setSelectedBet(null);
    setBetAmount('');
    setSelectedOutcome('');
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Welcome back, {user?.name}!</h2>
        <p className="mt-2 text-gray-600 dark:text-gray-300">Here's your betting overview</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Total Bets</h3>
          <p className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">{bets.length}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Won Bets</h3>
          <p className="text-3xl font-bold text-green-600">{bets.filter(bet => bet.status === 'Won').length}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Active Bets</h3>
          <p className="text-3xl font-bold text-yellow-600">{bets.filter(bet => bet.status === 'Pending').length}</p>
        </div>
      </div>

      {/* Place Bet Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Place a Bet</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Select Event</label>
              <select
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={selectedBet ? selectedBet.id : ''}
                onChange={(e) => setSelectedBet(availableBets.find(bet => bet.id === parseInt(e.target.value)))}
              >
                <option value="">Select an event</option>
                {availableBets.map(bet => (
                  <option key={bet.id} value={bet.id}>
                    {bet.event} - {bet.date}
                  </option>
                ))}
              </select>
            </div>

            {selectedBet && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Select Outcome</label>
                  <select
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                    value={selectedOutcome}
                    onChange={(e) => setSelectedOutcome(e.target.value)}
                  >
                    <option value="">Select outcome</option>
                    {Object.entries(selectedBet.odds).map(([key, value]) => (
                      <option key={key} value={key}>
                        {key.charAt(0).toUpperCase() + key.slice(1)} (Odds: {value})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Bet Amount</label>
                  <input
                    type="number"
                    min="1"
                    max={user.balance}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                    value={betAmount}
                    onChange={(e) => setBetAmount(e.target.value)}
                  />
                </div>

                <button
                  onClick={handlePlaceBet}
                  disabled={!betAmount || !selectedOutcome || parseFloat(betAmount) > user.balance}
                  className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 disabled:opacity-50"
                >
                  Place Bet
                </button>
              </>
            )}
          </div>

          {selectedBet && (
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">Bet Summary</h4>
              <div className="space-y-2 text-sm">
                <p className="text-gray-600 dark:text-gray-300">Event: {selectedBet.event}</p>
                <p className="text-gray-600 dark:text-gray-300">Date: {selectedBet.date}</p>
                <p className="text-gray-600 dark:text-gray-300">Time: {selectedBet.time}</p>
                {selectedOutcome && (
                  <>
                    <p className="text-gray-600 dark:text-gray-300">
                      Selected: {selectedOutcome.charAt(0).toUpperCase() + selectedOutcome.slice(1)}
                    </p>
                    <p className="text-gray-600 dark:text-gray-300">
                      Odds: {selectedBet.odds[selectedOutcome]}
                    </p>
                  </>
                )}
                {betAmount && (
                  <>
                    <p className="text-gray-600 dark:text-gray-300">Amount: ${betAmount}</p>
                    <p className="text-gray-600 dark:text-gray-300">
                      Potential Win: ${(parseFloat(betAmount) * (selectedBet.odds[selectedOutcome] || 0)).toFixed(2)}
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Bets Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Bets</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Odds</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Date</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {bets.map((bet) => (
                <tr key={bet.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">{bet.type}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">${bet.amount}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">{bet.odds}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <span className={`font-medium ${getStatusColor(bet.status)}`}>
                      {bet.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">{bet.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}