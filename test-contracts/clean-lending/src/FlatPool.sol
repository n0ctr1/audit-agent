// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0 ^0.8.19;

// lib/openzeppelin-contracts/contracts/token/ERC20/IERC20.sol

// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC20/IERC20.sol)

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the amount of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `from` to `to` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// lib/openzeppelin-contracts/contracts/security/ReentrancyGuard.sol

// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}

// src/CollateralLendingPool.sol

interface IOracle {
    // Returns the token price in USD with 18 decimals
    function getPrice(address token) external view returns (uint256);
}

contract CollateralLendingPool is ReentrancyGuard {
    IERC20 public stablecoin; // e.g. USDC
    IOracle public priceOracle;

    uint256 public constant COLLATERAL_RATIO = 150; // 150% collateralization
    uint256 public constant LIQUIDATION_THRESHOLD = 110; // Liquidation threshold 110%
    uint256 public constant LIQUIDATION_BONUS = 10; // Liquidator bonus 10%

    mapping(address => mapping(address => uint256)) public collateralBalance; // user -> token -> amount
    mapping(address => uint256) public borrowedAmount; // user -> stablecoin amount

    event Deposited(address indexed user, address indexed token, uint256 amount);
    event Borrowed(address indexed user, uint256 amount);
    event Liquidated(address indexed liquidator, address indexed user, address token, uint256 debtCovered);

    constructor(address _stablecoin, address _oracle) {
        stablecoin = IERC20(_stablecoin);
        priceOracle = IOracle(_oracle);
    }

    function depositCollateral(address token, uint256 amount) external nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        
        // Balance-before/after pattern to handle fee-on-transfer tokens
        uint256 balanceBefore = IERC20(token).balanceOf(address(this));
        require(IERC20(token).transferFrom(msg.sender, address(this), amount), "Transfer failed");
        uint256 actualAmount = IERC20(token).balanceOf(address(this)) - balanceBefore;

        collateralBalance[msg.sender][token] += actualAmount;
        emit Deposited(msg.sender, token, actualAmount);
    }

    function borrow(uint256 amount, address[] calldata collateralTokens) external nonReentrant {
        require(amount > 0, "Borrow amount must be > 0");
        
        uint256 totalCollateralValue = _getCollateralValue(msg.sender, collateralTokens);
        uint256 currentDebt = borrowedAmount[msg.sender];
        uint256 newDebt = currentDebt + amount;

        // Check: total collateral value must be >= 150% of the debt
        require((totalCollateralValue * 100) / newDebt >= COLLATERAL_RATIO, "Insufficient collateral");

        borrowedAmount[msg.sender] = newDebt;
        require(stablecoin.transfer(msg.sender, amount), "Transfer failed");
        
        emit Borrowed(msg.sender, amount);
    }

    function liquidate(
        address user, 
        address collateralToken, 
        uint256 debtToCover, 
        address[] calldata userAllCollaterals
    ) external nonReentrant {
        uint256 currentDebt = borrowedAmount[user];
        require(currentDebt > 0, "No active debt");
        require(debtToCover > 0 && debtToCover <= currentDebt, "Invalid debt amount");

        uint256 totalCollateralValue = _getCollateralValue(user, userAllCollaterals);
        
        // Check: collateralization has fallen below 110%
        require((totalCollateralValue * 100) / currentDebt <= LIQUIDATION_THRESHOLD, "Position is healthy");

        // Compute how much collateral goes to the liquidator
        uint256 tokenPrice = priceOracle.getPrice(collateralToken);
        uint256 collateralToTransfer = (debtToCover * 10**18) / tokenPrice;
        
        // Add the 10% bonus
        uint256 collateralWithBonus = collateralToTransfer + ((collateralToTransfer * LIQUIDATION_BONUS) / 100);

        require(collateralBalance[user][collateralToken] >= collateralWithBonus, "Not enough collateral to cover bonus");

        // Update state
        borrowedAmount[user] -= debtToCover;
        collateralBalance[user][collateralToken] -= collateralWithBonus;

        // Liquidator repays the debt
        require(stablecoin.transferFrom(msg.sender, address(this), debtToCover), "Debt repayment failed");
        // Liquidator receives the collateral
        require(IERC20(collateralToken).transfer(msg.sender, collateralWithBonus), "Collateral transfer failed");

        emit Liquidated(msg.sender, user, collateralToken, debtToCover);
    }

    // Internal helper: total value of a user's collateral
    function _getCollateralValue(address user, address[] calldata tokens) internal view returns (uint256) {
        uint256 totalValue = 0;
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 amount = collateralBalance[user][tokens[i]];
            if (amount > 0) {
                uint256 price = priceOracle.getPrice(tokens[i]);
                totalValue += (amount * price) / 10**18;
            }
        }
        return totalValue;
    }
}

